"""
trainer.py - End-to-end model benchmarking, Optuna HPO, and production serialization.
"""

import logging
import os
import sys
import json

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

from src.feature_store import TelcoFeaturePipeline
from src.evaluator import ModelEvaluator
from src.explainer import ModelExplainer

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("Trainer")


class ChurnModelTrainer:
    """Orchestrates model selection, Bayesian optimization, evaluation and artifact export."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.random_seed = self.config["project"]["random_seed"]
        self.target_col = self.config["features"]["target"]
        self.models_dir = self.config["paths"]["model_dir"]
        self.figures_dir = self.config["paths"]["figures_dir"]
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.pipeline: Optional[TelcoFeaturePipeline] = None
        self.best_model: Any = None
        self.evaluator = ModelEvaluator(self.figures_dir)

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Loads splits and executes feature engineering."""
        logger.info("Loading train, val, and test partitions...")
        train_df = pd.read_csv(self.config["paths"]["processed_train"])
        val_df = pd.read_csv(self.config["paths"]["processed_val"])
        test_df = pd.read_csv(self.config["paths"]["processed_test"])

        self.pipeline = TelcoFeaturePipeline(artifact_dir=self.models_dir)
        X_train = self.pipeline.fit_transform(train_df)
        X_val = self.pipeline.transform(val_df)
        X_test = self.pipeline.transform(test_df)

        y_train = train_df[self.target_col].values
        y_val = val_df[self.target_col].values
        y_test = test_df[self.target_col].values

        self.pipeline.save()
        return X_train, y_train, X_val, y_val, X_test, y_test, test_df

    def benchmark_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Compares 5 standard classification algorithms under Stratified 5-Fold CV."""
        logger.info("Running baseline model benchmark suite (Stratified 5-Fold CV)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)

        # Calculate negative/positive ratio for class balancing
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos = neg_count / float(pos_count)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.random_seed),
            "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=self.random_seed),
            "LightGBM": lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=self.random_seed, verbose=-1),
            "CatBoost": CatBoostClassifier(iterations=150, learning_rate=0.05, random_seed=self.random_seed, verbose=0),
            "XGBoost": xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, scale_pos_weight=scale_pos, eval_metric="logloss", random_state=self.random_seed)
        }

        results = {}
        for name, clf in models.items():
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            results[name] = float(np.mean(scores))
            logger.info(f"-> {name:<20}: Mean CV ROC-AUC = {results[name]:.4f} (+/- {np.std(scores):.4f})")

        return results

    def optimize_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 30) -> Dict[str, Any]:
        """Performs Bayesian hyperparameter optimization on XGBoost using Optuna."""
        logger.info(f"Starting Optuna Hyperparameter Optimization ({n_trials} trials)...")
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        base_scale_pos = neg_count / float(pos_count)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, base_scale_pos * 1.2),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": self.random_seed,
                "eval_metric": "logloss",
                "n_jobs": -1
            }
            clf = xgb.XGBClassifier(**params)
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Optuna completed. Best CV ROC-AUC: {study.best_value:.4f}")
        logger.info(f"Best Hyperparameters: {study.best_params}")
        return study.best_params

    def train_final_model(self, X_train_full: np.ndarray, y_train_full: np.ndarray, best_params: Dict[str, Any]) -> xgb.XGBClassifier:
        """Trains final XGBoost model with optimized hyperparameters."""
        logger.info("Training final production model on full training set...")
        best_params_clean = dict(best_params)
        best_params_clean["random_state"] = self.random_seed
        best_params_clean["eval_metric"] = "logloss"

        final_model = xgb.XGBClassifier(**best_params_clean)
        final_model.fit(X_train_full, y_train_full)
        self.best_model = final_model
        return final_model

    def run(self):
        """Executes full training pipeline, evaluation, and artifact generation."""
        X_train, y_train, X_val, y_val, X_test, y_test, test_df = self.load_and_prepare_data()

        # Step 1: Benchmark Models
        benchmark_results = self.benchmark_models(X_train, y_train)

        # Step 2: Optuna HPO
        n_trials = self.config["training"]["optuna_trials"]
        best_params = self.optimize_xgboost(X_train, y_train, n_trials=n_trials)

        # Step 3: Combine Train + Val for final model fit
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        model = self.train_final_model(X_train_full, y_train_full, best_params)

        # Step 4: Evaluate on completely held-out Test set (20% of data)
        logger.info("Evaluating final model on held-out test set...")
        y_test_proba = model.predict_proba(X_test)[:, 1]

        metrics = self.evaluator.compute_metrics(y_test, y_test_proba)
        logger.info("================ FINAL TEST METRICS ================")
        for k, v in metrics.items():
            logger.info(f"  {k:<15}: {v:.4f}")
        logger.info("====================================================")

        # Step 5: Slice-based Evaluation
        test_df_eval = test_df.copy()
        test_df_eval["actual_churn"] = y_test
        test_df_eval["predicted_churn_proba"] = y_test_proba
        slice_cols = ["Contract", "SeniorCitizen", "PaymentMethod", "InternetService"]
        slice_results = self.evaluator.compute_slice_metrics(
            test_df_eval, "actual_churn", "predicted_churn_proba", slice_cols
        )
        logger.info(f"Slice Performance: {json.dumps(slice_results, indent=2)}")

        # Step 6: Generate Diagnostic Plots
        roc_pr_plot = self.evaluator.plot_roc_pr_curves(y_test, y_test_proba, model_name="XGBoost_Optuna")
        cm_plot = self.evaluator.plot_confusion_matrix(y_test, y_test_proba, threshold=0.5, model_name="XGBoost_Optuna")

        # Step 7: SHAP Explainability
        logger.info("Generating SHAP explanations...")
        explainer = ModelExplainer(model, self.pipeline.feature_names, self.figures_dir)
        shap_values = explainer.explain_dataset(X_test, max_samples=400)
        shap_plot = explainer.plot_summary(shap_values)

        # Step 8: Save Model Artifacts & Evaluation Summary
        model_save_path = os.path.join(self.models_dir, "best_churn_model.pkl")
        joblib.dump(model, model_save_path)
        logger.info(f"Best model serialized to {model_save_path}")

        summary = {
            "benchmark_cv_auc": benchmark_results,
            "best_hyperparameters": best_params,
            "test_metrics": metrics,
            "slice_performance": slice_results,
            "artifacts": {
                "model": model_save_path,
                "feature_pipeline": os.path.join(self.models_dir, "feature_pipeline.joblib"),
                "roc_pr_curve": roc_pr_plot,
                "confusion_matrix": cm_plot,
                "shap_beeswarm": shap_plot
            }
        }

        summary_file = os.path.join(self.figures_dir, "model_results_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Execution summary written to {summary_file}")
        return summary


if __name__ == "__main__":
    trainer = ChurnModelTrainer()
    trainer.run()
