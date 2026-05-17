import os
import numpy as np
from joblib import Parallel, delayed
import rasterio
import logging
from tqdm import tqdm
import time
from utils import plot_classification_result, log_memory_usage
from feature_extraction import extract_pixel_features


def classify_image(sar_matrix, model, scaler, output_dir, transform,
                   window_size=3, batch_size=10000, n_jobs=-1):
    """Classify an entire SAR image pixel by pixel.

    Parameters
    ----------
    sar_matrix : np.ndarray
        2D SAR image array.
    model : sklearn classifier
        Trained model with predict() method.
    scaler : StandardScaler
        Fitted feature scaler.
    output_dir : str
    transform : affine.Affine
        Geotransform for writing the GeoTIFF.
    window_size : int
    batch_size : int
        Pixels per batch for memory efficiency.
    n_jobs : int
        Number of parallel feature extraction jobs.

    Returns
    -------
    predictions : np.ndarray
        2D array of class labels.
    """
    start_time = time.time()
    logging.info("Starting image classification...")

    indices = [(i, j) for i in range(sar_matrix.shape[0])
               for j in range(sar_matrix.shape[1])]
    predictions = np.zeros_like(sar_matrix, dtype=np.uint8)

    def process_batch(batch_indices):
        batch_features = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(extract_pixel_features)(i, j, sar_matrix, window_size)
            for i, j in batch_indices
        )
        batch_features = [f for f in batch_features if f is not None]
        if not batch_features:
            return np.zeros(len(batch_indices), dtype=np.uint8)
        batch_features = scaler.transform(batch_features)
        return model.predict(batch_features)

    try:
        total_batches = len(indices) // batch_size + (1 if len(indices) % batch_size else 0)
        for batch_start in tqdm(range(0, len(indices), batch_size),
                                total=total_batches, desc="Classifying batches",
                                ascii=True):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_predictions = process_batch(batch_indices)
            for (i, j), pred in zip(batch_indices[:len(batch_predictions)], batch_predictions):
                predictions[i, j] = pred

        pixel_counts = np.bincount(predictions.ravel())
        logging.info(f"Classification pixel distribution: {pixel_counts}")

        classified_raster = os.path.join(output_dir, "classified_result.tif")
        with rasterio.open(
            classified_raster, "w", driver="GTiff",
            height=predictions.shape[0], width=predictions.shape[1],
            count=1, dtype=rasterio.uint8, transform=transform
        ) as dst:
            dst.write(predictions, 1)
        logging.info(f"Classification raster saved to {classified_raster}")

        plot_classification_result(predictions, output_dir)

        logging.info(f"Image classification completed in {time.time() - start_time:.2f}s")
        log_memory_usage()
        return predictions
    except Exception as e:
        logging.error(f"Image classification failed: {e}")
        raise
