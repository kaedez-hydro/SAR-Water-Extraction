import os
import logging
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def setup_logging(output_dir):
    """Configure logging to file and console."""
    log_file = os.path.join(output_dir, 'process_log.txt')
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging setup complete. Memory usage: {psutil.virtual_memory().percent}%")


def log_memory_usage():
    """Log current memory usage."""
    memory_info = psutil.virtual_memory()
    logging.info(
        f"Memory usage: {memory_info.percent}%, "
        f"Used: {memory_info.used / 1024**3:.2f} GB, "
        f"Available: {memory_info.available / 1024**3:.2f} GB"
    )


def plot_confusion_matrix(cm, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def plot_classification_result(predictions, output_dir):
    """Plot and save classification result overview."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(predictions, cmap='viridis', cbar=True)
    plt.title('Classification Result')
    plt.savefig(os.path.join(output_dir, 'classification_result.png'))
    plt.close()
