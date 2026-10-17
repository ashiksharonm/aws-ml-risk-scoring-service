import json
import logging
import os
from pathlib import Path

import joblib
import xgboost as xgb
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.decisioning.model_card import build_model_card
from src.monitoring.drift import build_reference, save_reference
from src.training.data_loader import load_data
from src.training.preprocess import get_preprocessor, split_features_target
from src.utils.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train():
    logger.info("Loading data...")
    df = load_data()

    X, y = split_features_target(df)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    logger.info(f"Categorical features: {categorical_cols}")
    logger.info(f"Numerical features: {numerical_cols}")

    # Create Preprocessor
    preprocessor = get_preprocessor(categorical_cols, numerical_cols)

    # --- Baseline Model: Logistic Regression ---
    logger.info("Training Baseline Model (Logistic Integration)...")
    baseline_pipeline = Pipeline(
        [
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)

    # Evaluate Baseline
    y_pred_base = baseline_pipeline.predict(X_test)
    y_prob_base = baseline_pipeline.predict_proba(X_test)[:, 1]

    base_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_base)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_base)),
    }
    logger.info(f"Baseline Data: {base_metrics}")

    # --- Champion Model: XGBoost ---
    logger.info("Training Champion Model (XGBoost)...")
    # Note: XGBoost can handle missing values internally, but we use the preprocessor for consistency/OHE
    # For tree models, scaling is not strictly necessary but doesn't hurt.

    # We need to preprocess X separately for XGBoost if we want to use the sklearn pipeline structure effortlessly,
    # OR we can include XGBoost in the pipeline.
    champion_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                xgb.XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    eval_metric="auc",
                    # scale_pos_weight=1, # Revert to 1 to maximize Accuracy (user request)
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    champion_pipeline.fit(X_train, y_train)

    # Evaluate Champion
    y_pred_xgb = champion_pipeline.predict(X_test)
    y_prob_xgb = champion_pipeline.predict_proba(X_test)[:, 1]

    xgb_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_xgb)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_xgb)),
        "precision": float(classification_report(y_test, y_pred_xgb, output_dict=True)["1"]["precision"]),
        "recall": float(classification_report(y_test, y_pred_xgb, output_dict=True)["1"]["recall"]),
        "f1": float(classification_report(y_test, y_pred_xgb, output_dict=True)["1"]["f1-score"]),
    }

    logger.info(f"Champion XGBoost Metrics: {xgb_metrics}")

    # Save Metrics
    metrics = {
        "baseline": base_metrics,
        "champion": xgb_metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred_xgb).tolist(),
    }

    if not os.path.exists("reports"):
        os.makedirs("reports")

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save Artifacts
    if not os.path.exists("models"):
        os.makedirs("models")

    logger.info("Saving model artifacts...")
    joblib.dump(champion_pipeline, settings.MODEL_PATH)
    # The logistic-regression baseline doubles as the shadow-scored challenger.
    joblib.dump(baseline_pipeline, settings.CHALLENGER_PATH)

    # Training reference for PSI drift monitoring (features + champion score distribution).
    reference = build_reference(X_train, champion_pipeline.predict_proba(X_train)[:, 1])
    save_reference(reference, settings.REFERENCE_PATH)

    source = "synthetic" if os.getenv("USE_SYNTHETIC_DATA", "false").lower() == "true" else "uci"
    card = build_model_card(
        metrics, n_train=len(X_train), n_test=len(X_test), features=list(X.columns), data_source=source
    )
    Path(settings.MODEL_CARD_PATH).write_text(json.dumps(card, indent=2))

    logger.info("Training complete.")


if __name__ == "__main__":
    train()
