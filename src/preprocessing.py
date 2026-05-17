import os
import numpy as np
import rasterio
import logging


def load_sar_image(tif_path):
    """Load a SAR image and return the first band as a numpy array.

    Parameters
    ----------
    tif_path : str
        Path to the input GeoTIFF file.

    Returns
    -------
    matrix : np.ndarray
        2D array of pixel values.
    transform : affine.Affine
        Geotransform of the raster.
    """
    try:
        with rasterio.open(tif_path) as src:
            matrix = src.read(1)
            transform = src.transform
            height, width = src.height, src.width
        logging.info(f"Loaded SAR image: {height}x{width} pixels")
        return matrix, transform
    except Exception as e:
        logging.error(f"Failed to load SAR image: {e}")
        raise


def save_sar_as_matrix(tif_path, output_dir):
    """Load SAR image and save as .npy matrix for faster re-loading.

    Parameters
    ----------
    tif_path : str
        Path to the input GeoTIFF.
    output_dir : str
        Directory to save the .npy file.

    Returns
    -------
    matrix : np.ndarray
    transform : affine.Affine
    """
    matrix, transform = load_sar_image(tif_path)
    matrix_file = os.path.join(output_dir, "sar_matrix.npy")
    np.save(matrix_file, matrix)
    logging.info(f"SAR matrix saved to {matrix_file}")
    return matrix, transform
