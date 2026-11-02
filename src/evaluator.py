"""
evaluator.py - Google-grade model evaluation, slice-based error analysis, and metric visualization.
"""

import logging
import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("Evaluator")


class ModelEvaluator:
    """Rigorous evaluation suite including slice metrics and diagnostic plots."""

    def __init__(self, figures_dir: str = "reports/figures"):
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)

    def compute_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Calculates core classification metrics at specified decision threshold."""
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
            "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, y_pred_proba))
        }
        return metrics

    def compute_slice_metrics(self, df_with_preds: pd.DataFrame, target_col: str, proba_col: str, slice_cols: list) -> Dict[str, Dict[str, float]]:
        """
        Calculates slice-based metrics (Google ML fairness and sub-population performance).
        """
        slice_results = {}
        for col in slice_cols:
            if col not in df_with_preds.columns:
                continue
            slice_results[col] = {}
            for val, group in df_with_preds.groupby(col):
                if len(group[target_col].unique()) > 1:
                    auc = roc_auc_score(group[target_col], group[proba_col])
                    slice_results[col][str(val)] = round(float(auc), 4)
                else:
                    slice_results[col][str(val)] = None
        return slice_results

    def plot_roc_pr_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str = "XGBoost") -> str:
        """Generates ROC and Precision-Recall evaluation curves."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve
        axes[0].plot(fpr, tpr, color="#1a73e8", lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
        axes[0].plot([0, 1], [0, 1], color="#9aa0a6", linestyle="--", label="Chance")
        axes[0].set_title("ROC Curve", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)

        # PR Curve
        axes[1].plot(recall, precision, color="#d93025", lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
        baseline = np.mean(y_true)
        axes[1].axhline(y=baseline, color="#9aa0a6", linestyle="--", label=f"Baseline ({baseline:.3f})")
        axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(loc="lower left")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, f"{model_name.lower()}_evaluation_curves.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Evaluation curves saved to {output_path}")
        return output_path

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5, model_name: str = "XGBoost") -> str:
        """Visualizes confusion matrix with raw counts and normalized ratios."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm_norm, cmap="Blues", alpha=0.85)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center", color="black" if cm_norm[i, j] < 0.5 else "white",
                    fontsize=12, fontweight="bold"
                )

        fig.colorbar(cax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Retained (0)", "Churned (1)"])
        ax.set_yticklabels(["Retained (0)", "Churned (1)"])
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_title(f"Confusion Matrix ({model_name} @ th={threshold})", pad=20, fontweight="bold")

        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, f"{model_name.lower()}_confusion_matrix.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
