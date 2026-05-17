#!/usr/bin/env python3
"""
Local Binary Pattern (LBP) Feature Extraction
==============================================
Computes uniform LBP texture features for SAR imagery.
LBP captures local spatial structure by comparing each pixel
with its circular neighbors.

The resulting LBP image can be visualized as a heatmap
and saved for downstream analysis.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature
import cv2
from matplotlib.colors import LinearSegmentedColormap


def compute_lbp(image_path, P=8, R=1):
    """Compute uniform LBP for an image.

    Parameters
    ----------
    image_path : str
        Path to input image.
    P : int
        Number of circularly symmetric neighbors.
    R : float
        Radius of the circle.

    Returns
    -------
    lbp : np.ndarray
        LBP image (values depend on P and method).
    """
    image = io.imread(image_path)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = feature.local_binary_pattern(image, P, R, method="uniform")
    return lbp


def create_custom_colormap():
    """Create a smooth colormap for LBP visualization."""
    colors = [(0, 0, 1),          # Blue
              (0.2, 0.6, 1),      # Light Blue
              (0.5, 1, 0.5),      # Green
              (1, 0, 0)]          # Red
    positions = [0.0, 0.1, 0.5, 1.0]
    return LinearSegmentedColormap.from_list("custom_lbp",
                                              list(zip(positions, colors)))


def visualize_lbp(image_path, P=8, R=1, output_dir=None):
    """Compute, visualize, and optionally save LBP features.

    Parameters
    ----------
    image_path : str
    P : int
    R : float
    output_dir : str, optional
        Directory to save the LBP heatmap.
    """
    lbp = compute_lbp(image_path, P, R)
    custom_cmap = create_custom_colormap()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = ax.imshow(lbp, cmap=custom_cmap)
    plt.title("LBP Image", fontsize=15)
    plt.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("LBP Value", rotation=270, labelpad=15)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_lbp_heatmap.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved LBP heatmap: {output_path}")

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lbp_features.py <image_path> [P] [R] [output_dir]")
        print("Example: python lbp_features.py roi.tif 8 1 ./lbp_output")
        sys.exit(1)

    image_path = sys.argv[1]
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    R = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    output_dir = sys.argv[4] if len(sys.argv) > 4 else None

    visualize_lbp(image_path, P=P, R=R, output_dir=output_dir)
