"""
feature_store.py - Feature engineering, transformations, and artifact serialization.
Ensures zero data leakage: fitted on training set only.
"""

import logging
import os
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("FeatureStore")


class TelcoFeaturePipeline:
    """Production-grade feature transformer with state preservation for inference."""

    SERVICE_COLUMNS = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    NUMERICAL_BASE = ["tenure", "MonthlyCharges", "TotalCharges"]
    NUMERICAL_DERIVED = ["charges_per_tenure", "total_services", "charge_discrepancy"]
    ORDINAL_FEATURES = ["contract_risk_score"]

    CATEGORICAL_COLS = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "tenure_group"
    ]

    def __init__(self, artifact_dir: str = "models"):
        self.artifact_dir = artifact_dir
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies domain-specific feature engineering transformations."""
        data = df.copy()

        # 1. Monthly charge to tenure ratio (avoid division by 0)
        data["charges_per_tenure"] = data["MonthlyCharges"] / (data["tenure"] + 1.0)

        # 2. Total active service count
        service_presence = (data[self.SERVICE_COLUMNS] == "Yes").astype(int)
        data["total_services"] = service_presence.sum(axis=1)

        # 3. High risk contract mapping
        contract_map = {"Month-to-month": 3, "One year": 2, "Two year": 1}
        data["contract_risk_score"] = data["Contract"].map(contract_map).fillna(2)

        # 4. Tenure life-cycle segmentation
        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"]
        ).astype(str)

        # 5. Electronic Check payment flag (highest churn risk factor in dataset)
        data["is_electronic_check"] = (data["PaymentMethod"] == "Electronic check").astype(int)

        # 6. High-value senior flag
        data["is_senior_high_value"] = ((data["SeniorCitizen"] == 1) & (data["MonthlyCharges"] > 70.0)).astype(int)

        # 7. Customer retention protection factor (support + security)
        data["has_tech_support_or_security"] = (
            (data["TechSupport"] == "Yes") | (data["OnlineSecurity"] == "Yes")
        ).astype(int)

        # 8. Charge discrepancy (expected vs actual accumulated charges)
        expected_total = data["tenure"] * data["MonthlyCharges"]
        data["charge_discrepancy"] = data["TotalCharges"] - expected_total

        return data

    def fit(self, train_df: pd.DataFrame) -> "TelcoFeaturePipeline":
        """Fits scalers and encoders strictly on the training partition."""
        logger.info("Fitting feature pipeline on training dataset...")
        processed = self.engineer_features(train_df)

        numerical_features = self.NUMERICAL_BASE + self.NUMERICAL_DERIVED + self.ORDINAL_FEATURES + [
            "is_electronic_check", "is_senior_high_value", "has_tech_support_or_security"
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", RobustScaler(), numerical_features),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), self.CATEGORICAL_COLS)
            ],
            remainder="drop"
        )

        self.preprocessor.fit(processed)

        # Extract generated feature names
        num_names = numerical_features
        cat_names = self.preprocessor.named_transformers_["cat"].get_feature_names_out(self.CATEGORICAL_COLS).tolist()
        self.feature_names = num_names + cat_names

        logger.info(f"Feature pipeline fitted. Total transformed feature count: {len(self.feature_names)}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame into model-ready numpy array."""
        if self.preprocessor is None:
            raise RuntimeError("Pipeline must be fitted before transforming data.")
        processed = self.engineer_features(df)
        return self.preprocessor.transform(processed)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fits on train_df and returns transformed matrix."""
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: Optional[str] = None) -> str:
        """Serializes fitted pipeline to disk."""
        if filepath is None:
            Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)
            filepath = os.path.join(self.artifact_dir, "feature_pipeline.joblib")
        
        joblib.dump({"preprocessor": self.preprocessor, "feature_names": self.feature_names}, filepath)
        logger.info(f"Feature pipeline serialized to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str = "models/feature_pipeline.joblib") -> "TelcoFeaturePipeline":
        """Deserializes fitted pipeline from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Saved pipeline artifact not found at: {filepath}")
        
        payload = joblib.load(filepath)
        instance = cls()
        instance.preprocessor = payload["preprocessor"]
        instance.feature_names = payload["feature_names"]
        logger.info(f"Feature pipeline loaded successfully with {len(instance.feature_names)} features.")
        return instance
