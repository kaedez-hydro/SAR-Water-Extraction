"""
SHAP (SHapley Additive exPlanations) analysis for SAR water extraction models.

This module provides model interpretability through SHAP values,
helping understand which features (GLCM, LBP, Wavelet) contribute
most to water body classification decisions.
"""
import os
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def run_shap_analysis(model_path, scaler_path, features_path, labels_path,
                      output_dir, feature_names=None, sample_size=None):
    """Run comprehensive SHAP analysis on a trained model.

    Parameters
    ----------
    model_path : str
        Path to the saved model (.joblib or .json for XGBoost).
    scaler_path : str
        Path to the saved StandardScaler (.joblib).
    features_path : str
        Path to features.npy.
    labels_path : str
        Path to labels.npy.
    output_dir : str
        Directory for SHAP output figures.
    feature_names : list of str, optional
        Names of features. Auto-generated if None.
    sample_size : int, optional
        Number of background samples for explainer (use for large datasets).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    if model_path.endswith('.json'):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    else:
        model = joblib.load(model_path)

    scaler = joblib.load(scaler_path)

    features = np.load(features_path)
    labels = np.load(labels_path)

    if feature_names is None:
        n_glcm = 5
        n_lbp = 10
        n_wavelet = 8
        feature_names = (
            ["GLCM_contrast", "GLCM_dissimilarity", "GLCM_homogeneity",
             "GLCM_energy", "GLCM_correlation"][:n_glcm] +
            [f"LBP_{i}" for i in range(n_lbp)] +
            ["Wavelet_mean_cA", "mean_cH", "mean_cV", "mean_cD",
             "var_cA", "var_cH", "var_cV", "var_cD"][:n_wavelet]
        )

    X_scaled = scaler.transform(features)
    X_df = pd.DataFrame(X_scaled, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, labels, test_size=0.2, random_state=42
    )

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Multi-class or binary
    if isinstance(shap_values, list):
        class_names = [f"Class_{i}" for i in range(len(shap_values))]
    else:
        class_names = ["Class_0"]

    # ---- Summary plots per class ----
    for idx, class_name in enumerate(class_names):
        vals = shap_values[idx] if isinstance(shap_values, list) else shap_values
        plt.figure()
        shap.summary_plot(vals, X_test, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary - {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"summary_{class_name}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ---- Bar plot (overall importance) ----
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("Overall Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_bar_all_classes.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Stacked bar by class ----
    if isinstance(shap_values, list):
        mean_shap = [np.abs(vals).mean(axis=0) for vals in shap_values]
        stacked = np.array(mean_shap)
        total = np.sum(stacked, axis=0)
        sorted_idx = np.argsort(total)[::-1]

        fig, ax = plt.subplots(figsize=(12, 6))
        left = np.zeros(len(feature_names))
        colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
        for i, cname in enumerate(class_names):
            vals = stacked[i][sorted_idx]
            ax.barh(np.array(feature_names)[sorted_idx], vals,
                    left=left, label=cname, color=colors[i])
            left += vals
        ax.invert_yaxis()
        ax.set_xlabel("Mean(|SHAP Value|)")
        ax.set_title("Stacked Feature Importance by Class")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "stacked_importance.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"SHAP analysis complete. Figures saved to: {output_dir}")
    return shap_values, X_test
