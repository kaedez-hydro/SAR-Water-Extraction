import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump
import logging
from tqdm import tqdm
import time
from utils import plot_confusion_matrix, log_memory_usage


def train_random_forest(X_train, y_train, X_val, y_val, features_scaled, labels,
                         output_dir, n_estimators=100, max_depth=10, n_jobs=6):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        n_jobs=n_jobs, random_state=42
    )
    with tqdm(total=100, desc="Training Random Forest", ascii=True) as pbar:
        model.fit(X_train, y_train)
        pbar.update(100)

    dump(model, os.path.join(output_dir, "random_forest_model.joblib"))

    cv_scores = cross_val_score(model, features_scaled, labels, cv=5, n_jobs=n_jobs)
    logging.info(f"RF 5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    y_pred = model.predict(X_val)
    logging.info(f"RF Validation accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    logging.info("RF Classification report:\n" + classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, output_dir)

    return model


def train_svm(X_train, y_train, X_val, y_val, features_scaled, labels,
              output_dir, kernel='rbf', C=1.0):
    """Train an SVM classifier."""
    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    with tqdm(total=100, desc="Training SVM", ascii=True) as pbar:
        model.fit(X_train, y_train)
        pbar.update(100)

    dump(model, os.path.join(output_dir, "svm_model.joblib"))

    cv_scores = cross_val_score(model, features_scaled, labels, cv=5, n_jobs=-1)
    logging.info(f"SVM 5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    y_pred = model.predict(X_val)
    logging.info(f"SVM Validation accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    logging.info("SVM Classification report:\n" + classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, output_dir)

    return model


def train_xgboost(X_train, y_train, X_val, y_val, features_scaled, labels,
                  output_dir, n_estimators=100, max_depth=10, learning_rate=0.1,
                  use_gpu=False):
    """Train an XGBoost classifier."""
    import xgboost as xgb

    tree_method = 'gpu_hist' if use_gpu else 'hist'
    predictor = 'gpu_predictor' if use_gpu else 'cpu_predictor'

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method=tree_method,
        predictor=predictor,
        random_state=42
    )

    with tqdm(total=100, desc="Training XGBoost", ascii=True) as pbar:
        model.fit(X_train, y_train)
        pbar.update(100)

    model.save_model(os.path.join(output_dir, "xgboost_model.json"))

    y_pred = model.predict(X_val)
    logging.info(f"XGBoost Validation accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    logging.info("XGBoost Classification report:\n" + classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, output_dir)

    return model


MODEL_FACTORY = {
    'rf': train_random_forest,
    'random_forest': train_random_forest,
    'svm': train_svm,
    'xgboost': train_xgboost,
    'xgb': train_xgboost,
}


def train_and_validate(features, labels, output_dir, model_type='rf', **model_kwargs):
    """Train and validate a classification model.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Label vector.
    output_dir : str
        Directory for saving outputs.
    model_type : str
        One of 'rf', 'svm', 'xgboost'.
    **model_kwargs
        Additional arguments passed to the model constructor.

    Returns
    -------
    model : trained classifier
    scaler : StandardScaler
    """
    start_time = time.time()
    logging.info(f"Starting {model_type.upper()} model training...")
    logging.info(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    if not np.issubdtype(labels.dtype, np.integer):
        logging.warning(f"Labels dtype is {labels.dtype}, converting to int64")
        if not np.all(labels == labels.astype(int)):
            raise ValueError("Labels contain non-integer values")
        labels = labels.astype(np.int64)
    logging.info(f"Label distribution: {np.bincount(labels)}")

    if features.shape[0] != labels.shape[0]:
        raise ValueError("Features and labels have mismatched shapes")

    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        dump(scaler, os.path.join(output_dir, "scaler.joblib"))

        X_train, X_val, y_train, y_val = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42
        )
        logging.info(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")
        logging.info(f"Train label distribution: {np.bincount(y_train)}")
        logging.info(f"Validation label distribution: {np.bincount(y_val)}")

        train_fn = MODEL_FACTORY.get(model_type)
        if train_fn is None:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(MODEL_FACTORY.keys())}")

        model = train_fn(
            X_train, y_train, X_val, y_val,
            features_scaled, labels, output_dir,
            **model_kwargs
        )

        logging.info(f"Model training completed in {time.time() - start_time:.2f}s")
        log_memory_usage()
        return model, scaler
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise
