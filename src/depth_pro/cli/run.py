#!/usr/bin/env python3
"""Sample script to run DepthPro with automatic PLY output."""

import argparse
import logging
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from depth_pro import create_model_and_transforms, load_rgb

LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def visualize_2d(image, depth, fig, ax_rgb, ax_disp, output_file=None):
    """2D visualization of RGB and depth using matplotlib."""
    # Input validation
    if depth.min() <= 0:
        raise ValueError("Depth values must be positive")
    
    # Calculate normalized inverse depth for visualization
    inverse_depth = 1 / depth
    max_invdepth = min(inverse_depth.max(), 1 / 0.1)  # Cap at 0.1m
    min_invdepth = max(1 / 250, inverse_depth.min())  # Cap at 250m
    inverse_depth_normalized = (inverse_depth - min_invdepth) / (max_invdepth - min_invdepth)
    
    # Display the image and estimated depth map
    ax_rgb.clear()
    ax_disp.clear()
    
    # RGB visualization
    ax_rgb.imshow(image)
    ax_rgb.set_title('RGB Image')
    ax_rgb.axis('off')
    
    # Depth visualization
    depth_vis = ax_disp.imshow(inverse_depth_normalized, cmap="turbo")
    ax_disp.set_title('Depth Map')
    ax_disp.axis('off')
    
    # Add colorbar
    fig.colorbar(depth_vis, ax=ax_disp, label='Inverse Depth')
    
    # Save if output path provided
    if output_file is not None:
        # Convert to uint8 and apply colormap
        depth_img = (inverse_depth_normalized * 255).clip(0, 255).astype(np.uint8)
        
        # Apply the same colormap as in the plot
        cmap = plt.get_cmap('turbo')
        colored_depth = (cmap(depth_img) * 255).astype(np.uint8)
        
        # The cmap returns RGBA, but we want RGB for JPEG
        colored_depth_rgb = colored_depth[:, :, :3]
        
        PIL.Image.fromarray(colored_depth_rgb).save(str(output_file) + ".jpg", format="JPEG", quality=90)
    
    # Update figure
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

def run(args):
    """Run Depth Pro on a sample image."""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load model
    model, transform = create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    # Set up input paths
    image_paths = [args.image_path]
    if args.image_path.is_dir():
        image_paths = args.image_path.glob("**/*")
        relative_path = args.image_path
    else:
        relative_path = args.image_path.parent

    # Create default output directory
    output_dir = Path("./models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize matplotlib for visualization
    plt.ion()
    fig = plt.figure()
    ax_rgb = fig.add_subplot(121)
    ax_disp = fig.add_subplot(122)

    for image_path in tqdm(image_paths):
        # Skip non-image files
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        try:
            LOGGER.info(f"Processing {image_path} ...")
            image, _, f_px = load_rgb(image_path)
        except Exception as e:
            LOGGER.error(f"Error loading {image_path}: {str(e)}")
            continue
            
        # Run prediction
        prediction = model.infer(transform(image), f_px=f_px)
        depth = prediction["depth"]
        depth_np = depth.detach().cpu().numpy().squeeze()

        # Show 2D visualization
        rel_path = image_path.relative_to(relative_path)
        depth_output = output_dir / rel_path.parent / image_path.stem
        visualize_2d(image, depth_np, fig, ax_rgb, ax_disp, depth_output)
        
        # Create output path maintaining directory structure
        output_path = output_dir / rel_path.parent / f"{image_path.stem}.ply"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export PLY with our enhanced function
        LOGGER.info(f"Saving PLY to: {output_path}")
        pcd = model.export_to_ply(
            depth=depth,
            rgb_image=image,
            focallength_px=prediction["focallength_px"],
            output_path=str(output_path),
            voxel_size=args.voxel_size,
            estimate_normals=True
        )

        # Show 3D visualization
        if not args.no_display:
            LOGGER.info("Displaying point cloud...")
            model.display_point_cloud(
                pcd,
                window_name=f"DepthPro - {image_path.name}",
                point_size=2.0
            )

    LOGGER.info("Processing complete!")
    
    # Keep matplotlib window open
    if not args.no_display:
        plt.show(block=True)


def main():
    """Run DepthPro inference with automatic PLY export."""
    parser = argparse.ArgumentParser(
        description="DepthPro inference with automatic PLY export"
    )
    parser.add_argument(
        "-i", 
        "--image-path", 
        type=Path, 
        default="./data/example.jpg",
        help="Path to input image or directory",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel size for point cloud downsampling (default: 0.01)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip displaying visualizations",
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Show verbose output",
    )
    
    run(parser.parse_args())


if __name__ == "__main__":
    main()