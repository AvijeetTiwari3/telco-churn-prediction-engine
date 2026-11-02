"""
test_api.py - Unit and integration tests for FastAPI Churn inference service.
"""

import pytest
from pydantic import ValidationError
from api.app import health_check, predict_churn, predict_batch, CustomerInput


def test_health_check_function():
    res = health_check()
    assert "status" in res
    assert res["status"] in ["HEALTHY", "INITIALIZING"]
    assert res["version"] == "1.0.0"


def test_customer_input_validation_error():
    # SeniorCitizen must be 0 or 1
    with pytest.raises(ValidationError):
        CustomerInput(
            customerID="INVALID",
            gender="Female",
            SeniorCitizen=5,  # Invalid
            Partner="No",
            Dependents="No",
            tenure=-1,  # Invalid
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="DSL",
            OnlineSecurity="No",
            OnlineBackup="No",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="No",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=50.0,
            TotalCharges=50.0
        )


def test_predict_churn_single_customer():
    payload = CustomerInput(
        customerID="TEST-CHURN-01",
        gender="Female",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=2,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=85.50,
        TotalCharges=171.00
    )

    result = predict_churn(payload)
    assert result.customerID == "TEST-CHURN-01"
    assert 0.0 <= result.churn_probability <= 1.0
    assert result.risk_level in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(result.top_risk_drivers, list)
    assert len(result.top_risk_drivers) > 0
    assert "direction" in result.top_risk_drivers[0]
    assert "impact" in result.top_risk_drivers[0]


def test_predict_batch_customers():
    cust1 = CustomerInput(
        customerID="BATCH-01",
        gender="Male",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="Yes",
        tenure=48,
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Two year",
        PaperlessBilling="No",
        PaymentMethod="Credit card (automatic)",
        MonthlyCharges=65.0,
        TotalCharges=3120.0
    )
    cust2 = CustomerInput(
        customerID="BATCH-02",
        gender="Female",
        SeniorCitizen=1,
        Partner="No",
        Dependents="No",
        tenure=1,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=75.0,
        TotalCharges=75.0
    )

    batch_res = predict_batch([cust1, cust2])
    assert len(batch_res) == 2
    assert batch_res[0].customerID == "BATCH-01"
    assert batch_res[1].customerID == "BATCH-02"
    # Long tenure, two-year contract should have lower churn probability than 1-month fiber customer
    assert batch_res[0].churn_probability < batch_res[1].churn_probability
