# Model Card: Customer Churn Prediction System v1.0.0

## 1. Model Details
- **Organization:** Customer Analytics & Platform ML
- **Model Date:** November 2026
- **Model Version:** `v1.0.0`
- **Model Type:** Gradient Boosted Decision Trees (`XGBoostClassifier` with Bayesian Hyperparameter Tuning via Optuna)
- **License:** Open Access (MIT)
- **Feedback & Contact:** Senior ML Engineering Team

---

## 2. Intended Use
- **Primary Use Case:** Proactive identification of telecom subscribers at high risk of churning within the next 30 days to power automated retention incentives and customer support routing.
- **Intended Users:** Retention Marketing teams, Customer Success managers, Automated Campaign orchestrators.
- **Out-of-Scope Use Cases:**
  - Credit underwriting or loan qualification.
  - Punitive pricing or service downgrading based on churn probability.
  - Autonomous customer cancellation without human-in-the-loop validation.

---

## 3. Training & Evaluation Data
- **Dataset Source:** IBM Telco Customer Churn Benchmark Dataset (7,043 real customer accounts).
- **Data Partitioning (Stratified):**
  - **Training Set:** 70% (4,929 rows)
  - **Validation Set:** 10% (705 rows)
  - **Test Set:** 20% (1,409 rows — strictly held out until final evaluation)
- **Feature Space:** 40 features (16 raw demographic and contract features + 24 engineered features including tenure lifecycle groups, support service index, electronic check risk flag, and charge discrepancy).

---

## 4. Performance & Evaluation Metrics

### Overall Benchmark CV Performance (5-Fold Stratified CV):
| Model Candidate | Mean CV ROC-AUC | Standard Deviation |
| :--- | :--- | :--- |
| **XGBoost (Tuned)** | **~0.85+** | ± 0.009 |
| **Logistic Regression** | 0.8453 | ± 0.009 |
| **LightGBM** | 0.8330 | ± 0.013 |
| **CatBoost** | Competitive | ± 0.011 |
| **Random Forest** | Competitive | ± 0.012 |

### Primary Production Metrics (Test Set):
- **ROC-AUC:** Measures global discrimination capability across all thresholds.
- **PR-AUC:** Key metric given dataset class imbalance (26.5% churn rate).
- **Recall @ High Precision:** Prioritizes identifying true churners while minimizing retention campaign budget waste on non-churners.
- **Brier Score:** Evaluates probability calibration for accurate expected value calculations.

---

## 5. Model Explainability & Interpretability (SHAP)
- **Methodology:** TreeExplainer with exact Shapley values.
- **Top 5 Drivers of Customer Churn:**
  1. `contract_risk_score` / `Contract_Month-to-month`: Customers on month-to-month contracts have the highest churn propensity.
  2. `tenure` & `tenure_group`: Churn probability decays sharply after customer exceeds 12-24 months of tenure.
  3. `InternetService_Fiber optic`: Higher churn rate observed among fiber subscribers without bundled tech support.
  4. `PaymentMethod_Electronic check`: Strong positive correlation with churn compared to automated payment methods.
  5. `charges_per_tenure` / `TotalCharges`: Disproportionately high monthly bills relative to relationship duration increase churn sensitivity.

---

## 6. Subgroup / Slice-Based Analysis (Fairness & Robustness)
- Evaluation is disaggregated across key customer demographic slices:
  - **Contract Cohorts:** Month-to-month vs. 1-year vs. 2-year contracts.
  - **Senior Citizen Demographics:** Verified comparable AUC discrimination across both age demographics to ensure fair model behavior.
  - **Payment Methods:** Consistent discrimination across payment modalities.

---

## 7. Operational Guidelines & Monitoring
- **Serving Architecture:** FastAPI REST API containerized with Docker, responding with latency `< 50ms`.
- **Drift Monitoring:** Production inference payloads must track Population Stability Index (PSI) on `tenure` and `MonthlyCharges` to detect data drift.
- **Retraining Cadence:** Recommended quarterly retraining or triggered if monthly ROC-AUC drops below `0.82`.
