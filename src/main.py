#!/usr/bin/env python3
"""
SAR Water Body Extraction Pipeline
===================================
End-to-end workflow for water body classification from SAR imagery
using GLCM + LBP + Wavelet features with Random Forest / SVM / XGBoost.

Usage:
    python main.py                          # Run with default config.yaml
    python main.py --config path/to.yaml    # Custom config
    python main.py --model svm              # Override model type
    python main.py --model xgboost          # Use XGBoost
"""
import os
import sys
import argparse
import yaml
import logging
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from utils import setup_logging, log_memory_usage
from preprocessing import save_sar_as_matrix
from feature_extraction import extract_features_parallel
from model_training import train_and_validate
from classification import classify_image


def load_config(config_path):
    """Load YAML configuration and resolve relative paths."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        project_root = os.path.dirname(os.path.abspath(__file__))
        for key in ['tif_path', 'label_shp_path', 'output_dir']:
            config['paths'][key] = os.path.abspath(
                os.path.join(project_root, config['paths'][key])
            )
        return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise


def validate_inputs(tif_path, label_shp_path):
    """Verify input files exist."""
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"TIFF file not found: {tif_path}")
    if not os.path.exists(label_shp_path):
        raise FileNotFoundError(f"Shapefile not found: {label_shp_path}")
    logging.info("Input validation passed.")


def main():
    parser = argparse.ArgumentParser(
        description="SAR Water Body Extraction Pipeline"
    )
    parser.add_argument('--config', default='config.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--model', default=None,
                        choices=['rf', 'svm', 'xgboost'],
                        help='Override model type from config')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               args.config if os.path.isabs(args.config)
                               else args.config)
    if not os.path.exists(config_path):
        # Try relative to CWD
        config_path = args.config
    config = load_config(config_path)
    paths = config['paths']
    params = config['parameters']

    # Override model if specified
    model_type = args.model or params.get('model_type', 'rf')

    # Setup
    setup_logging(paths['output_dir'])
    log_memory_usage()

    # Validate inputs
    validate_inputs(paths['tif_path'], paths['label_shp_path'])

    # Step 1: Load SAR image
    sar_matrix, transform = save_sar_as_matrix(
        paths['tif_path'], paths['output_dir']
    )

    # Step 2: Load labels
    labels_gdf = gpd.read_file(paths['label_shp_path'])
    logging.info(f"Loaded Shapefile with {len(labels_gdf)} geometries")

    if 'label' not in labels_gdf.columns:
        raise ValueError("Shapefile must have a 'label' column")
    labels_gdf['label'] = labels_gdf['label'].astype(np.int64)
    logging.info(f"Label distribution: {np.bincount(labels_gdf['label'])}")

    # Step 3: Feature extraction
    features, labels = extract_features_parallel(
        sar_matrix, labels_gdf, transform,
        window_size=params.get('window_size', 3),
        n_jobs=params.get('n_jobs', 6),
        max_pixels=params.get('max_pixels', 1000)
    )

    # Step 4: Model training
    model, scaler = train_and_validate(
        features, labels, paths['output_dir'],
        model_type=model_type
    )

    # Step 5: Full image classification
    predictions = classify_image(
        sar_matrix, model, scaler, paths['output_dir'], transform,
        window_size=params.get('window_size', 3),
        batch_size=params.get('batch_size', 10000),
        n_jobs=params.get('n_jobs', 6)
    )

    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program failed: {e}")
        raise
    finally:
        plt.close('all')
