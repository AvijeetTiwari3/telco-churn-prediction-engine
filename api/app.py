"""
app.py - Production-ready FastAPI REST service for real-time Churn scoring and SHAP explainability.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.feature_store import TelcoFeaturePipeline
from src.explainer import ModelExplainer

app = FastAPI(
    title="Customer Churn Prediction & Explainability Engine",
    description="Enterprise-grade inference API for predicting telecom customer churn with SHAP explanations.",
    version="1.0.0"
)

START_TIME = time.time()
MODEL_PATH = "models/best_churn_model.pkl"
PIPELINE_PATH = "models/feature_pipeline.joblib"

# Global references loaded at startup
model: Optional[Any] = None
pipeline: Optional[TelcoFeaturePipeline] = None
explainer: Optional[ModelExplainer] = None


class CustomerInput(BaseModel):
    customerID: Optional[str] = Field(default="CUST-0001", description="Unique customer identifier")
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=100, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0.0, example=85.5)
    TotalCharges: float = Field(..., ge=0.0, example=1026.0)


class ChurnPredictionResponse(BaseModel):
    customerID: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    decision_threshold: float
    top_risk_drivers: List[Dict[str, Any]]
    model_version: str


@app.on_event("startup")
def load_artifacts():
    global model, pipeline, explainer
    if os.path.exists(MODEL_PATH) and os.path.exists(PIPELINE_PATH):
        model = joblib.load(MODEL_PATH)
        pipeline = TelcoFeaturePipeline.load(PIPELINE_PATH)
        explainer = ModelExplainer(model, pipeline.feature_names)
    else:
        # Will be loaded once training finishes
        pass


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health status and metadata verification."""
    is_ready = model is not None and pipeline is not None
    return {
        "status": "HEALTHY" if is_ready else "INITIALIZING",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "model_loaded": is_ready,
        "model_path": MODEL_PATH,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=ChurnPredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerInput):
    """Predicts churn probability and explains key drivers for a single customer."""
    global model, pipeline, explainer
    if model is None or pipeline is None:
        load_artifacts()
        if model is None or pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts are not ready yet. Please ensure trainer.py has run."
            )

    # Convert customer dictionary to DataFrame
    customer_dict = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
    df = pd.DataFrame([customer_dict])

    # Transform features
    x_matrix = pipeline.transform(df)

    # Model inference
    churn_prob = float(model.predict_proba(x_matrix)[0, 1])
    is_churn = bool(churn_prob >= 0.5)

    # Risk categorization
    if churn_prob >= 0.70:
        risk_level = "HIGH"
    elif churn_prob >= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # SHAP explanations
    top_drivers = explainer.explain_single_prediction(x_matrix[0], top_k=4)

    return ChurnPredictionResponse(
        customerID=customer.customerID or "UNKNOWN",
        churn_probability=round(churn_prob, 4),
        churn_prediction=is_churn,
        risk_level=risk_level,
        decision_threshold=0.50,
        top_risk_drivers=top_drivers,
        model_version="1.0.0"
    )


@app.post("/predict/batch", response_model=List[ChurnPredictionResponse], tags=["Inference"])
def predict_batch(customers: List[CustomerInput]):
    """Batch inference endpoint for high-throughput batch scoring."""
    results = []
    for customer in customers:
        results.append(predict_churn(customer))
    return results
