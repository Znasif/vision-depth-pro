#!/usr/bin/env python3
"""Streamlit server for Depth Pro."""

import streamlit as st
import torch
import tempfile
import numpy as np
from PIL import Image
import logging
import depth_pro
from depth_pro.cli.run import get_torch_device
import base64
import av
import open3d as o3d
import io
import time
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'frame_idx' not in st.session_state:
    st.session_state.frame_idx = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False

def get_point_cloud_data(depth: np.ndarray, rgb_image: np.ndarray, focal_length: float, voxel_size: float = None):
    """Convert depth and RGB data to point cloud format."""
    height, width = depth.shape
    
    # Create coordinate grid
    x_grid, y_grid = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )
    
    # Convert image coordinates to 3D points
    Z = depth
    X = (x_grid - width/2) * Z / focal_length
    Y = (y_grid - height/2) * Z / focal_length
    
    # Stack coordinates and reshape
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3) / 255.0
    
    # Create Open3D point cloud for processing
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Remove invalid points
    pcd = pcd.remove_non_finite_points()
    
    # Optional downsampling
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)
    
    # Convert back to numpy arrays
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    return points, colors

def extract_frames(video_file: io.BytesIO, sample_rate: int = 1):
    """Extract frames from video using PyAV."""
    container = av.open(video_file)
    stream = container.streams.video[0]
    
    for frame_idx, frame in enumerate(container.decode(stream)):
        if frame_idx % sample_rate == 0:
            yield frame.to_image()
    
    container.close()

def get_video_info(video_file: io.BytesIO):
    """Get video metadata using PyAV."""
    container = av.open(video_file)
    stream = container.streams.video[0]
    
    total_frames = stream.frames
    fps = float(stream.average_rate)
    width = stream.width
    height = stream.height
    
    container.close()
    
    return total_frames, fps, (width, height)

def save_frames_as_video(frames, output_path: str, fps: float):
    """Save frames as video using PyAV."""
    if not frames:
        return
        
    container = av.open(output_path, mode='w')
    stream = container.add_stream('h264', rate=fps)
    stream.width = frames[0].width
    stream.height = frames[0].height
    stream.pix_fmt = 'yuv420p'
    
    for img in frames:
        frame = av.VideoFrame.from_image(img)
        packet = stream.encode(frame)
        container.mux(packet)
    
    # Flush stream
    packet = stream.encode(None)
    container.mux(packet)
    container.close()

# Initialize model
@st.cache_resource
def init_model():
    device = get_torch_device()
    try:
        model, transform = depth_pro.create_model_and_transforms(
            device=device,
            precision=torch.half,
            weights_only=True
        )
    except TypeError:
        model, transform = depth_pro.create_model_and_transforms(
            device=device,
            precision=torch.half,
        )
    model.eval()
    return model, transform

def get_threejs_code():
    """Get the Three.js visualization code."""
    return """
    <script type="importmap">
        {
            "imports": {
                "three": "https://unpkg.com/three@0.157.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.157.0/examples/jsm/"
            }
        }
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        class PointCloudViewer {
            constructor(container) {
                this.container = container;
                this.init();
            }

            init() {
                // Setup scene
                this.scene = new THREE.Scene();
                this.scene.background = new THREE.Color(0xf0f0f0);

                // Setup camera
                this.camera = new THREE.PerspectiveCamera(
                    75,
                    this.container.clientWidth / this.container.clientHeight,
                    0.1,
                    1000
                );
                this.camera.position.z = 5;

                // Setup renderer
                this.renderer = new THREE.WebGLRenderer({ antialias: true });
                this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
                this.container.appendChild(this.renderer.domElement);

                // Setup controls
                this.controls = new OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;

                // Add coordinate frame
                const axesHelper = new THREE.AxesHelper(2);
                this.scene.add(axesHelper);

                // Start animation loop
                this.animate();
            }

            updatePointCloud(points, colors) {
                // Remove existing point cloud if any
                if (this.pointCloud) {
                    this.scene.remove(this.pointCloud);
                    this.pointCloud.geometry.dispose();
                    this.pointCloud.material.dispose();
                }

                // Create geometry
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

                // Create material
                const material = new THREE.PointsMaterial({
                    size: 0.02,
                    vertexColors: true,
                    sizeAttenuation: true
                });

                // Create point cloud
                this.pointCloud = new THREE.Points(geometry, material);
                this.scene.add(this.pointCloud);

                // Adjust camera
                const box = new THREE.Box3().setFromObject(this.pointCloud);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = this.camera.fov * (Math.PI / 180);
                const cameraDistance = Math.abs(maxDim / Math.sin(fov / 2));

                this.camera.position.copy(center);
                this.camera.position.z += cameraDistance;
                this.camera.lookAt(center);

                this.controls.target.copy(center);
                this.controls.update();
            }

            animate = () => {
                requestAnimationFrame(this.animate);
                this.controls.update();
                this.renderer.render(this.scene, this.camera);
            }
        }

        // Initialize viewer
        const container = document.getElementById('point-cloud-container');
        const viewer = new PointCloudViewer(container);

        // Get point cloud data from window object
        const points = window.pointCloudData.points;
        const colors = window.pointCloudData.colors;
        viewer.updatePointCloud(points, colors);
    </script>
    """

def create_point_cloud_html(points, colors):
    """Create HTML content for point cloud visualization."""
    point_cloud_data = {
        "points": points.tolist(),
        "colors": colors.tolist()
    }
    
    html_content = f"""
    <div id="point-cloud-container" style="height: 400px; width: 100%; position: relative;">
        <div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; font-size: 12px;">
            Left click + drag: Rotate<br>
            Right click + drag: Pan<br>
            Scroll: Zoom
        </div>
    </div>
    <script>
        window.pointCloudData = {json.dumps(point_cloud_data)};
    </script>
    {get_threejs_code()}
    """
    
    return html_content

def main():
    st.title("Depth Pro - 3D Reconstruction")
    
    # Initialize model
    model, transform = init_model()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    voxel_size = st.sidebar.slider(
        "Point Cloud Density", 
        min_value=0.001, 
        max_value=0.1, 
        value=0.01,
        help="Lower values create denser point clouds but increase processing time"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=["jpg", "jpeg", "png", "mp4"],
        key="depth_pro_uploader"
    )
    
    if uploaded_file is not None:
        # Check if it's a video file
        is_video = uploaded_file.name.lower().endswith('.mp4')
        
        if is_video:
            frame_sampling = st.sidebar.slider(
                "Frame Sampling Rate", 
                min_value=1, 
                max_value=30, 
                value=5,
                help="Process every Nth frame"
            )
        
        # Process button
        if st.button("Process File"):
            try:
                if is_video:
                    # Process video
                    video_bytes = io.BytesIO(uploaded_file.read())
                    total_frames, fps, (width, height) = get_video_info(video_bytes)
                    
                    # Create temporary files for output
                    with tempfile.NamedTemporaryFile(suffix='_depth.mp4', delete=False) as depth_file:
                        # Reset video pointer
                        video_bytes.seek(0)
                        
                        # Process frames
                        progress_bar = st.progress(0)
                        depth_frames = []
                        processed_count = 0
                        
                        for i, frame in enumerate(extract_frames(video_bytes, frame_sampling)):
                            try:
                                # Convert to RGB if needed
                                if frame.mode != 'RGB':
                                    frame = frame.convert('RGB')
                                
                                # Process image
                                prediction = model.infer(transform(np.array(frame)))
                                depth = prediction["depth"].cpu().numpy()
                                
                                # Create colored depth visualization
                                import matplotlib.pyplot as plt
                                depth_colored = Image.fromarray(
                                    (plt.cm.turbo((depth - depth.min()) / (depth.max() - depth.min())) * 255
                                ).astype(np.uint8)[..., :3])
                                depth_frames.append(depth_colored)
                                
                                processed_count += 1
                                progress_bar.progress((i * frame_sampling + 1) / total_frames)
                                
                            except Exception as e:
                                st.error(f"Error processing frame {i}: {str(e)}")
                                continue
                        
                        # Save depth video
                        save_frames_as_video(depth_frames, depth_file.name, fps/frame_sampling)
                        
                        # Provide download button
                        st.success(f"Processed {processed_count} frames successfully!")
                        
                        with open(depth_file.name, 'rb') as f:
                            depth_bytes = f.read()
                        st.download_button(
                            label="Download Depth Video",
                            data=depth_bytes,
                            file_name="depth_video.mp4",
                            mime="video/mp4"
                        )
                
                else:
                    # Process single image
                    image = Image.open(uploaded_file)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Display original image
                    st.image(image, caption="Input Image", use_column_width=True)
                    
                    # Process image
                    prediction = model.infer(transform(np.array(image)))
                    depth = prediction["depth"]
                    
                    # Create point cloud data
                    points, colors = get_point_cloud_data(
                        depth.cpu().numpy(),
                        np.array(image),
                        prediction["focallength_px"].item(),
                        voxel_size
                    )
                    
                    # Create columns for visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Depth Map")
                        # Normalize depth map for visualization
                        depth_viz = (depth.cpu().numpy() - depth.cpu().numpy().min()) / (
                            depth.cpu().numpy().max() - depth.cpu().numpy().min()
                        )
                        st.image(depth_viz, caption="Depth Map", use_column_width=True)
                        
                    with col2:
                        st.subheader("3D Point Cloud")
                        html_content = create_point_cloud_html(points, colors)
                        st.components.v1.html(html_content, height=400)
                    
                    # Show success message
                    st.success("Processing complete!")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                logger.exception("Error in processing")

if __name__ == "__main__":
    main()