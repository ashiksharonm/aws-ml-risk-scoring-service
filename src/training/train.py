import joblib
import json
import logging
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import pandas as pd
import numpy as np

from src.utils.config import settings
from src.training.data_loader import load_data
from src.training.preprocess import get_preprocessor, split_features_target

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train():
    logger.info("Loading data...")
    df = load_data()
    
    X, y = split_features_target(df)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Categorical features: {categorical_cols}")
    logger.info(f"Numerical features: {numerical_cols}")
    
    # Create Preprocessor
    preprocessor = get_preprocessor(categorical_cols, numerical_cols)
    
    # --- Baseline Model: Logistic Regression ---
    logger.info("Training Baseline Model (Logistic Integration)...")
    baseline_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
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
    champion_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    champion_pipeline.fit(X_train, y_train)
    
    # Evaluate Champion
    y_pred_xgb = champion_pipeline.predict(X_test)
    y_prob_xgb = champion_pipeline.predict_proba(X_test)[:, 1]
    
    xgb_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_xgb)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_xgb)),
        "precision": float(classification_report(y_test, y_pred_xgb, output_dict=True)['1']['precision']),
        "recall": float(classification_report(y_test, y_pred_xgb, output_dict=True)['1']['recall']),
        "f1": float(classification_report(y_test, y_pred_xgb, output_dict=True)['1']['f1-score'])
    }
    
    logger.info(f"Champion XGBoost Metrics: {xgb_metrics}")
    
    # Save Metrics
    metrics = {
        "baseline": base_metrics,
        "champion": xgb_metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred_xgb).tolist()
    }
    
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Artifacts
    if not os.path.exists("models"):
        os.makedirs("models")
        
    logger.info("Saving model and preprocessor...")
    # Saving the full pipeline for easy inference
    joblib.dump(champion_pipeline, settings.MODEL_PATH)
    
    # We might want to save just the preprocessor or just the model sometimes, 
    # but saving the pipeline is best for production.
    # However, for SHAP, we often need the raw model and transformed data.
    # Let's also save the preprocessor separately if needed, but pipeline has it named 'preprocessor'.
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
