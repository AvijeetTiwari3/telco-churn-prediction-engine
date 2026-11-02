"""
data_pipeline.py - Data ingestion, schema validation, and dataset splitting.
Author: Google-grade Industry ML Pipeline
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("DataPipeline")


class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass


class TelcoDataPipeline:
    """Production-grade pipeline for loading, validating, and splitting Telco data."""

    EXPECTED_COLUMNS = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
    ]

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.raw_data_path = self.config["paths"]["raw_data"]
        self.train_path = self.config["paths"]["processed_train"]
        self.val_path = self.config["paths"]["processed_val"]
        self.test_path = self.config["paths"]["processed_test"]
        self.target_col = self.config["features"]["target"]
        self.random_seed = self.config["project"]["random_seed"]

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Safely loads YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_raw_data(self) -> pd.DataFrame:
        """Loads raw Telco CSV and validates file presence."""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data file not found at: {self.raw_data_path}")
        
        logger.info(f"Loading raw dataset from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validates dataframe schema against expected columns and types."""
        logger.info("Executing schema validation checks...")
        
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns in dataset: {missing_cols}")
        
        if df.empty:
            raise DataValidationError("Dataset is empty.")
            
        logger.info("Schema validation passed successfully.")
        return True

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data types and addresses known edge-cases in the IBM Telco dataset:
        1. TotalCharges contains empty strings ' ' for new customers with tenure = 0.
        2. Encode binary Churn column (Yes -> 1, No -> 0).
        """
        logger.info("Starting raw data cleaning...")
        df = df.copy()

        # Handle TotalCharges blank strings -> convert to float
        empty_mask = df["TotalCharges"].astype(str).str.strip() == ""
        num_empty = empty_mask.sum()
        if num_empty > 0:
            logger.info(f"Detected {num_empty} blank entries in TotalCharges (tenure=0). Imputing with 0.0.")
            df.loc[empty_mask, "TotalCharges"] = "0.0"

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        # Standardize Target variable: Yes -> 1, No -> 0
        if self.target_col in df.columns:
            if df[self.target_col].dtype == object:
                df[self.target_col] = df[self.target_col].map({"Yes": 1, "No": 0})
                if df[self.target_col].isnull().any():
                    raise DataValidationError("Target column contains values other than 'Yes' and 'No'")

        logger.info(f"Data cleaning complete. Target distribution: \n{df[self.target_col].value_counts(normalize=True).to_dict()}")
        return df

    def split_and_save(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset into stratified Train (70%), Validation (10%), and Test (20%).
        Saves clean splits to disk.
        """
        logger.info("Splitting dataset into stratified Train / Validation / Test sets...")
        test_size = self.config["data_split"]["test_size"]
        val_size = self.config["data_split"]["val_size"]

        # First split off the test set (20%)
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=df[self.target_col]
        )

        # Split remaining 80% into Train (70% total) and Val (10% total) -> val is 10/80 = 0.125 of train_val
        val_relative_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=self.random_seed,
            stratify=train_val_df[self.target_col]
        )

        logger.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Ensure target directories exist
        for path in [self.train_path, self.val_path, self.test_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(self.train_path, index=False)
        val_df.to_csv(self.val_path, index=False)
        test_df.to_csv(self.test_path, index=False)

        logger.info(f"Splits saved successfully to: {self.train_path}, {self.val_path}, {self.test_path}")
        return train_df, val_df, test_df

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Orchestrates end-to-end data pipeline."""
        df = self.load_raw_data()
        self.validate_schema(df)
        df_clean = self.clean_raw_data(df)
        return self.split_and_save(df_clean)


if __name__ == "__main__":
    pipeline = TelcoDataPipeline()
    pipeline.run()
