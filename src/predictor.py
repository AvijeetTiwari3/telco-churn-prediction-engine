"""
predictor.py - Production inference engine for customer churn prediction.
Supports single record scoring and bulk batch inference from CSV.
"""

import os
import sys
import argparse
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_store import TelcoFeaturePipeline
from src.explainer import ModelExplainer


class ChurnPredictor:
    """Inference wrapper combining model artifacts, feature pipeline, and SHAP explanations."""

    def __init__(self, model_path: str = "models/best_churn_model.pkl", pipeline_path: str = "models/feature_pipeline.joblib"):
        if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
            raise FileNotFoundError("Model or feature pipeline artifact not found. Run trainer.py first.")

        self.model = joblib.load(model_path)
        self.pipeline = TelcoFeaturePipeline.load(pipeline_path)
        self.explainer = ModelExplainer(self.model, self.pipeline.feature_names)

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Runs single-record inference with risk scoring and SHAP feature drivers."""
        df = pd.DataFrame([record])
        x_mat = self.pipeline.transform(df)
        prob = float(self.model.predict_proba(x_mat)[0, 1])

        risk_level = "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.35 else "LOW"
        top_drivers = self.explainer.explain_single_prediction(x_mat[0], top_k=4)

        return {
            "customerID": record.get("customerID", "UNKNOWN"),
            "churn_probability": round(prob, 4),
            "churn_prediction": bool(prob >= 0.50),
            "risk_level": risk_level,
            "top_drivers": top_drivers
        }

    def predict_batch_csv(self, input_csv: str, output_csv: str, threshold: float = 0.50) -> pd.DataFrame:
        """Loads input CSV, executes predictions, and exports scored customer list."""
        df = pd.read_csv(input_csv)
        x_mat = self.pipeline.transform(df)
        probs = self.model.predict_proba(x_mat)[:, 1]

        df["churn_probability"] = np.round(probs, 4)
        df["churn_prediction"] = (probs >= threshold).astype(int)
        df["risk_level"] = np.where(probs >= 0.70, "HIGH", np.where(probs >= 0.35, "MEDIUM", "LOW"))

        df.to_csv(output_csv, index=False)
        return df


def main():
    parser = argparse.ArgumentParser(description="Telecom Customer Churn Inference CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, default="data/scored_customers.csv", help="Path to save predictions")
    args = parser.parse_args()

    predictor = ChurnPredictor()
    results = predictor.predict_batch_csv(args.input, args.output)
    print(f"Scored {len(results)} customers successfully. Saved to {args.output}")


if __name__ == "__main__":
    main()
