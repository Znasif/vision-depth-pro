#!/usr/bin/env python3
"""Streamlit server for Depth Pro."""

import streamlit as st
import torch
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import logging
import depth_pro
from depth_pro.cli.run import get_torch_device
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
@st.cache_resource
def init_model():
    device = get_torch_device()
    model, transform = depth_pro.create_model_and_transforms(
        device=device,
        precision=torch.half,
    )
    model.eval()
    return model, transform

def process_image(image_data, model, transform, voxel_size=None):
    """Process an image and return the PLY file path and visualization."""
    try:
        # Convert to RGB if necessary
        if image_data.mode != 'RGB':
            image_data = image_data.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image_data)
        
        # Load and process the image
        prediction = model.infer(transform(image_np))
        depth = prediction["depth"]
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.ply"
            
            # Export to PLY
            pcd = model.export_to_ply(
                depth=depth,
                rgb_image=image_np,
                focallength_px=prediction["focallength_px"],
                output_path=str(output_path),
                voxel_size=voxel_size,
                estimate_normals=True
            )
            
            # Read the PLY file
            with open(output_path, 'rb') as f:
                ply_data = f.read()
            
            return ply_data, depth.cpu().numpy(), prediction["focallength_px"].item()
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate HTML for file download."""
    b64 = base64.b64encode(bin_file).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_label}">Download {file_label}</a>'

def main():
    st.title("Depth Pro - 3D Reconstruction")
    
    # Initialize model
    model, transform = init_model()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    voxel_size = st.sidebar.slider(
        "Point Cloud Density (voxel size)", 
        min_value=0.001, 
        max_value=0.1, 
        value=0.01,
        help="Lower values create denser point clouds but increase file size"
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", use_column_width=True)
        
        # Process button
        if st.button("Process Image"):
            with st.spinner("Processing image..."):
                try:
                    # Process image
                    ply_data, depth_map, focal_length = process_image(image, model, transform, voxel_size)
                    
                    # Create columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Depth Map")
                        # Normalize depth map for visualization
                        depth_viz = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                        st.image(depth_viz, caption="Depth Map", use_column_width=True, clamp=True)
                        
                    with col2:
                        st.subheader("Download")
                        st.markdown(get_binary_file_downloader_html(ply_data, 'output.ply'), unsafe_allow_html=True)
                        st.info(f"Estimated Focal Length: {focal_length:.2f} pixels")
                    
                    # Additional information
                    st.success("Processing complete! You can now download the PLY file.")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    logger.exception("Error in processing")

if __name__ == "__main__":
    main()