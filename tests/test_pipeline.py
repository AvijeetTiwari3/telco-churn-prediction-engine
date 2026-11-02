"""
test_pipeline.py - Automated tests for data pipeline and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_pipeline import TelcoDataPipeline, DataValidationError
from src.feature_store import TelcoFeaturePipeline


@pytest.fixture
def sample_raw_df():
    """Generates realistic sample dataframe mimicking IBM Telco dataset."""
    return pd.DataFrame({
        "customerID": ["001-TEST", "002-TEST", "003-TEST"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 24, 0],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 89.5, 20.0],
        "TotalCharges": ["29.85", "2148.0", " "],  # Includes known blank TotalCharges case
        "Churn": ["No", "No", "Yes"]
    })


def test_schema_validation_success(sample_raw_df):
    pipeline = TelcoDataPipeline()
    assert pipeline.validate_schema(sample_raw_df) is True


def test_schema_validation_failure_on_missing_column(sample_raw_df):
    pipeline = TelcoDataPipeline()
    dropped_df = sample_raw_df.drop(columns=["Contract"])
    with pytest.raises(DataValidationError):
        pipeline.validate_schema(dropped_df)


def test_clean_raw_data_handles_whitespace_total_charges(sample_raw_df):
    pipeline = TelcoDataPipeline()
    cleaned = pipeline.clean_raw_data(sample_raw_df)
    
    # Verify whitespace converted to 0.0 float
    assert cleaned.loc[2, "TotalCharges"] == 0.0
    assert cleaned["TotalCharges"].dtype in [np.float64, np.float32, float]
    
    # Verify Churn mapped to binary {0, 1}
    assert list(cleaned["Churn"].values) == [0, 0, 1]


def test_feature_engineering_pipeline(sample_raw_df):
    pipeline = TelcoDataPipeline()
    cleaned = pipeline.clean_raw_data(sample_raw_df)

    feat_pipeline = TelcoFeaturePipeline()
    X = feat_pipeline.fit_transform(cleaned)

    # Check matrix properties
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 3
    assert X.shape[1] == len(feat_pipeline.feature_names)
    assert not np.isnan(X).any(), "Transformed feature matrix contains NaNs!"
