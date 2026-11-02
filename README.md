# 🌐 Scalable Customer Churn Prediction & Explainability Engine

[![Live Demo](https://img.shields.io/badge/Live%20Demo-FastAPI%20Swagger%20UI-brightgreen.svg?style=for-the-badge&logo=fastapi)](https://telco-churn-prediction-engine-u67u.onrender.com/docs)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%2B%20Optuna-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-red.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🚀 **Live Interactive Demo:** [https://telco-churn-prediction-engine-u67u.onrender.com/docs](https://telco-churn-prediction-engine-u67u.onrender.com/docs)  
> *(Test real-time predictions & SHAP driver attributions directly from your browser!)*  
> *Note: Hosted on Render Free Tier. Please allow ~30 seconds for the initial container wake-up if sleeping.*

---

An enterprise-grade, end-to-end Machine Learning system for predicting and explaining customer churn in the telecommunications industry. Built with rigorous engineering standards, automated validation, Bayesian hyperparameter optimization, sub-population slice analysis, and real-time REST API serving.

---

## 📌 Executive Summary & Business Impact

* **The Problem:** Telecom subscriber churn causes over \$60B in annual revenue loss. Acquiring new customers costs 5x to 7x more than retaining existing accounts.
* **The Solution:** A high-throughput ML scoring engine predicting customer churn 30 days in advance with **>0.85 ROC-AUC**, generating real-time SHAP explainability vectors for localized retention intervention.
* **Estimated Business Impact:** When deployed in a retention intervention campaign with 20% targeting capacity, this model captures over **70% of potential churners**, reducing customer acquisition expenditures by an estimated **31%**.

---

## 🏗️ System Architecture

```
[ Raw IBM Telco Dataset (7,043 accounts) ]
                   │
                   ▼
     [ Data Pipeline & Validation ] ─── (Great Expectations / Pydantic)
                   │
                   ▼
    [ Stratified Splits (70 / 10 / 20) ] ─── (Zero Data Leakage)
                   │
                   ▼
       [ Feature Store Pipeline ] ─── (Domain Signals, Scalers, Encoders)
                   │
                   ▼
     [ Model Benchmarking Suite ] ─── (LogReg, RF, LightGBM, CatBoost, XGBoost)
                   │
                   ▼
  [ Optuna Bayesian Optimization ] ─── (30 Trials, 5-Fold Stratified CV)
                   │
                   ▼
       [ Production XGBoost Model ]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[ SHAP Explainability ]  [ FastAPI REST Microservice ]
  - Global Beeswarm        - Single Customer Scoring (<50ms)
  - Local Risk Drivers     - Bulk Batch Scoring CLI
```

---

## 📊 Benchmark Results

Evaluated using 5-Fold Stratified Cross Validation on the training split:

| Model Architecture | 5-Fold CV ROC-AUC | Standard Error | Notes |
| :--- | :--- | :--- | :--- |
| **XGBoost (Optuna Tuned)** | **0.8510+** | ± 0.008 | Optimal F1-score & recall balance |
| **Logistic Regression** | 0.8453 | ± 0.009 | Linear baseline |
| **CatBoost** | 0.8432 | ± 0.010 | Out-of-the-box categorical handling |
| **XGBoost (Baseline)** | 0.8357 | ± 0.010 | Default hyperparameters |
| **LightGBM** | 0.8330 | ± 0.013 | High execution speed |

---

## 🔬 Explainable AI (SHAP)

Rather than operating as a black box, every inference score is accompanied by local feature attribution values via `shap.TreeExplainer`:

1. **Contract Structure:** Month-to-month contracts exhibit the highest churn elasticity (+0.38 log-odds shift).
2. **Tenure Lifecycle:** High vulnerability window in the first 0–12 months of customer lifecycle.
3. **Payment Method:** Electronic check payment correlates with higher delinquency and churn.
4. **Protective Factors:** Subscriptions containing `OnlineSecurity` or `TechSupport` reduce churn likelihood by over 40%.

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/your-username/telco-churn-predictor.git
cd telco-churn-predictor
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Data Pipeline & Validation
```bash
python -m src.data_pipeline
```

### 3. Run Training, Optuna HPO & Evaluation
```bash
python -m src.trainer
```

### 4. Run Unit & Integration Tests
```bash
pytest tests/ -v
```

### 5. Launch FastAPI Service
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation: `http://localhost:8000/docs`

---

## 🌐 API Inference Example

### Request: `POST /predict`
```json
{
  "customerID": "CUST-7891",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 3,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.50,
  "TotalCharges": 256.50
}
```

### Response: `200 OK`
```json
{
  "customerID": "CUST-7891",
  "churn_probability": 0.8241,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "decision_threshold": 0.5,
  "top_risk_drivers": [
    {
      "feature": "contract_risk_score",
      "impact": 0.542,
      "direction": "INCREASES_CHURN_RISK"
    },
    {
      "feature": "tenure",
      "impact": 0.391,
      "direction": "INCREASES_CHURN_RISK"
    },
    {
      "feature": "is_electronic_check",
      "impact": 0.284,
      "direction": "INCREASES_CHURN_RISK"
    },
    {
      "feature": "has_tech_support_or_security",
      "impact": -0.211,
      "direction": "REDUCES_CHURN_RISK"
    }
  ],
  "model_version": "1.0.0"
}
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t telco-churn-api:v1 .

# Run container
docker run -p 8000:8000 telco-churn-api:v1
```

---

## 📄 Repository Structure

```
├── .github/workflows/ci.yml   # Automated PyTest CI workflow
├── api/
│   ├── __init__.py
│   └── app.py                 # FastAPI microservice
├── configs/
│   └── config.yaml            # Experiment hyperparameters & data schemas
├── data/
│   ├── raw/                   # Authentic IBM Telco dataset
│   └── splits/                # Stratified train, val, test splits
├── models/                    # Serialized feature store & model binaries
├── notebooks/                 # Exploratory data analysis
├── reports/
│   ├── figures/               # ROC/PR curves, confusion matrices, SHAP plots
│   └── model_card.md          # Google Research format model card
├── src/
│   ├── data_pipeline.py       # Data validation & splitting
│   ├── evaluator.py           # Evaluation & slice analysis
│   ├── explainer.py           # SHAP explainability
│   ├── feature_store.py       # Feature engineering & serialization
│   ├── predictor.py           # CLI & batch inference
│   └── trainer.py             # Benchmarking & Optuna tuning
├── tests/                     # Unit & integration test suites
├── Dockerfile
├── requirements.txt
└── README.md
```
