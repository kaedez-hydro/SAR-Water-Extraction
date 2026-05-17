#!/usr/bin/env python3
"""
Wavelet Decomposition Feature Visualization
============================================
Performs 2D Discrete Wavelet Transform (DWT) on a grayscale image
and visualizes the four sub-bands: Approximation (cA), Horizontal
detail (cH), Vertical detail (cV), and Diagonal detail (cD).

Useful for understanding the multi-scale texture information
used as features in the SAR water extraction pipeline.
"""
import os
import sys
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt


def plot_wavelet_components(image_path, wavelet='haar', output_dir=None, cmap='viridis'):
    """Extract and visualize 2D wavelet decomposition components.

    Parameters
    ----------
    image_path : str
        Path to input image (supports common formats: JPG, PNG, TIFF).
    wavelet : str
        Wavelet type ('haar', 'db2', 'db4', 'sym4', etc.).
    output_dir : str, optional
        If provided, save component images to this directory.
    cmap : str
        Matplotlib colormap name.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Cannot read image: {image_path}")
        return

    image = image.astype(np.float32) / 255.0

    coeffs = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs

    titles = ['Approximation (Low Frequency)', 'Horizontal Detail',
              'Vertical Detail', 'Diagonal Detail']
    components = [cA, cH, cV, cD]

    plt.figure(figsize=(12, 8))
    for i, (title, comp) in enumerate(zip(titles, components)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(comp, cmap=cmap)
        plt.title(title)
        plt.axis('off')

        if output_dir:
            safe_name = title.replace(' ', '_').replace('(', '').replace(')', '')
            comp_output = os.path.join(output_dir, f"{safe_name}.jpg")
            plt.imsave(comp_output, comp, cmap=cmap)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wavelet_features.py <image_path> [wavelet_type] [output_dir]")
        print("Example: python wavelet_features.py roi.tif haar ./wavelet_output")
        sys.exit(1)

    image_path = sys.argv[1]
    wavelet = sys.argv[2] if len(sys.argv) > 2 else 'haar'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    plot_wavelet_components(image_path, wavelet=wavelet, output_dir=output_dir, cmap='cividis')
