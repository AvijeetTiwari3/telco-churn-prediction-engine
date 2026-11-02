"""
explainer.py - SHAP model explainability for global transparency and local customer scoring.
"""

import logging
import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

# Fix XGBoost 3.x ubjson base_score formatting compatibility with SHAP
try:
    _orig_decode = shap.explainers._tree.decode_ubjson_buffer

    def _patched_decode_ubjson_buffer(fd):
        jmodel = _orig_decode(fd)
        try:
            bs = jmodel["learner"]["learner_model_param"]["base_score"]
            if isinstance(bs, str):
                jmodel["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
        except Exception:
            pass
        return jmodel

    shap.explainers._tree.decode_ubjson_buffer = _patched_decode_ubjson_buffer
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("Explainer")


class ModelExplainer:
    """Computes and visualizes SHAP explanations for tree-based models."""

    def __init__(self, model: Any, feature_names: List[str], figures_dir: str = "reports/figures"):
        self.model = model
        self.feature_names = feature_names
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_dataset(self, X_sample: np.ndarray, max_samples: int = 500) -> shap.Explanation:
        """Calculates SHAP values for a sample dataset."""
        if X_sample.shape[0] > max_samples:
            indices = np.random.choice(X_sample.shape[0], max_samples, replace=False)
            X_eval = X_sample[indices]
        else:
            X_eval = X_sample

        logger.info(f"Computing SHAP values across {X_eval.shape[0]} samples...")
        shap_values = self.explainer(X_eval)
        shap_values.feature_names = self.feature_names
        return shap_values

    def plot_summary(self, shap_values: shap.Explanation) -> str:
        """Plots global feature importance summary beeswarm."""
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.title("SHAP Feature Importance (Impact on Churn Prediction)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, "shap_summary_beeswarm.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP beeswarm saved to {out_path}")
        return out_path

    def explain_single_prediction(self, x_vector: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Explains an individual customer's prediction for real-time API responses.
        Returns top positive and negative risk factors.
        """
        if x_vector.ndim == 1:
            x_vector = x_vector.reshape(1, -1)

        raw_shap = self.explainer.shap_values(x_vector)
        # Handle binary classification output format
        if isinstance(raw_shap, list):
            vals = raw_shap[1][0]
        elif raw_shap.ndim == 3:
            vals = raw_shap[0, :, 1]
        else:
            vals = raw_shap[0]

        factors = []
        for i, val in enumerate(vals):
            factors.append({
                "feature": self.feature_names[i],
                "impact": float(val),
                "direction": "INCREASES_CHURN_RISK" if val > 0 else "REDUCES_CHURN_RISK"
            })

        # Sort by absolute magnitude of impact
        factors.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return factors[:top_k]
