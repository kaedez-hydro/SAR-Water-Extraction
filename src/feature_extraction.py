import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import geopandas as gpd
from rasterio.features import geometry_mask
from joblib import Parallel, delayed
import logging
from tqdm import tqdm
import time


def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """Extract GLCM texture features from an image window.

    Returns
    -------
    list
        [contrast, dissimilarity, homogeneity, energy, correlation]
    """
    try:
        image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        glcm = graycomatrix(image_uint8, distances=distances, angles=angles,
                            levels=levels, symmetric=True, normed=True)
        features = [
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'dissimilarity').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'correlation').mean(),
        ]
        return features
    except Exception as e:
        logging.warning(f"GLCM extraction failed: {e}")
        return [0.0] * 5


def extract_lbp_features(image, radius=1, n_points=8):
    """Extract LBP histogram features from an image window.

    Returns
    -------
    list
        LBP histogram (uniform pattern, n_points+2 bins).
    """
    try:
        image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        lbp = local_binary_pattern(image_uint8, P=n_points, R=radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2), density=True)
        return hist.tolist()
    except Exception as e:
        logging.warning(f"LBP extraction failed: {e}")
        return [0.0] * (n_points + 2)


def extract_wavelet_features(image, wavelet='haar'):
    """Extract wavelet decomposition features from an image window.

    Returns
    -------
    list
        [mean(cA), mean(cH), mean(cV), mean(cD),
         var(cA), var(cH), var(cV), var(cD)]
    """
    try:
        coeffs = pywt.dwt2(image, wavelet)
        cA, (cH, cV, cD) = coeffs
        features = [
            cA.mean(), cH.mean(), cV.mean(), cD.mean(),
            cA.var(),  cH.var(),  cV.var(),  cD.var()
        ]
        return features
    except Exception as e:
        logging.warning(f"Wavelet extraction failed: {e}")
        return [0.0] * 8


def extract_pixel_features(i, j, sar_matrix, window_size=3):
    """Extract combined GLCM+LBP+Wavelet features for a single pixel.

    Parameters
    ----------
    i, j : int
        Pixel coordinates.
    sar_matrix : np.ndarray
        Full SAR image matrix.
    window_size : int
        Size of the square window around the pixel.

    Returns
    -------
    list or None
        Concatenated feature vector [GLCM(5) + LBP(10) + Wavelet(8)] = 23 features.
    """
    half_window = window_size // 2
    row_start, row_end = max(0, i - half_window), min(sar_matrix.shape[0], i + half_window + 1)
    col_start, col_end = max(0, j - half_window), min(sar_matrix.shape[1], j + half_window + 1)
    window = sar_matrix[row_start:row_end, col_start:col_end]

    if window.size < 2:
        return None

    try:
        glcm_features = extract_glcm_features(window)
        lbp_features = extract_lbp_features(window)
        wavelet_features = extract_wavelet_features(window)
        return glcm_features + lbp_features + wavelet_features
    except Exception as e:
        logging.warning(f"Feature extraction failed for pixel ({i}, {j}): {e}")
        return None


def get_feature_names():
    """Return the list of feature names."""
    glcm_names = ["GLCM_contrast", "GLCM_dissimilarity", "GLCM_homogeneity",
                  "GLCM_energy", "GLCM_correlation"]
    lbp_names = [f"LBP_{i}" for i in range(10)]
    wavelet_names = ["Wavelet_mean_cA", "mean_cH", "mean_cV", "mean_cD",
                     "var_cA", "var_cH", "var_cV", "var_cD"]
    return glcm_names + lbp_names + wavelet_names


def extract_features_parallel(sar_matrix, labels_gdf, transform,
                               window_size=3, n_jobs=6, max_pixels=1000):
    """Extract features in parallel from labelled geometries.

    Parameters
    ----------
    sar_matrix : np.ndarray
    labels_gdf : GeoDataFrame
        Labeled polygons with a 'label' column.
    transform : affine.Affine
    window_size : int
    n_jobs : int
        Number of parallel jobs.
    max_pixels : int
        Maximum pixels to sample per geometry (subsampling).

    Returns
    -------
    features_array : np.ndarray of shape (n_samples, n_features)
    labels_array : np.ndarray of shape (n_samples,)
    """
    start_time = time.time()

    def process_geometry(row):
        mask = geometry_mask(
            [row.geometry],
            transform=transform,
            invert=True,
            out_shape=sar_matrix.shape
        )
        indices = np.argwhere(mask)
        logging.info(f"Geometry {row.Index}: {len(indices)} pixels")
        if len(indices) > max_pixels:
            logging.info(f"Subsampling geometry {row.Index} to {max_pixels} pixels")
            indices = indices[np.random.choice(len(indices), max_pixels, replace=False)]
        features = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(extract_pixel_features)(i, j, sar_matrix, window_size)
            for i, j in indices
        )
        valid = [f for f in features if f is not None]
        return valid, [row.label] * len(valid)

    try:
        results = []
        for row in tqdm(labels_gdf.itertuples(), total=len(labels_gdf),
                        desc="Processing geometries", ascii=True):
            feats, lbls = process_geometry(row)
            results.append((feats, lbls))

        all_features, all_labels = [], []
        for feats, lbls in results:
            all_features.extend(feats)
            all_labels.extend(lbls)
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)

        if features_array.size == 0:
            raise ValueError("No features extracted.")

        logging.info(f"Feature extraction completed in {time.time() - start_time:.2f}s")
        return features_array, labels_array
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        raise
