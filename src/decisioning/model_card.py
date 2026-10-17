"""Model card assembled from the metrics produced by a training run."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from src.decisioning.reason_codes import EXCLUDED_FEATURES


def build_model_card(metrics: dict, n_train: int, n_test: int, features: List[str], data_source: str) -> dict:
    synthetic = data_source == "synthetic"
    return {
        "model": "XGBoost binary classifier (champion), logistic regression (challenger)",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "intended_use": "Portfolio/demo of credit-default scoring with explainable decisions. "
        "Not validated for real lending decisions.",
        "training_data": {
            "source": "synthetic random data (NOT meaningful)"
            if synthetic
            else "UCI Default of Credit Card Clients (Taiwan, 2005)",
            "train_rows": n_train,
            "test_rows": n_test,
            "split": "80/20 stratified, seed 42",
            "features": features,
        },
        "metrics_on_holdout": {
            "champion": metrics["champion"],
            "challenger_baseline": metrics["baseline"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "explainability": "TreeSHAP log-odds contributions; adverse-action reasons use the "
        "top positive contributors merged into reason codes.",
        "excluded_from_reason_codes": sorted(EXCLUDED_FEATURES),
        "limitations": [
            "Single-country, single-year dataset; no out-of-time validation.",
            "Default 0.5 threshold gives low recall on defaults; tune per cost of errors.",
            "The model still takes sex, age, marriage and education as inputs; they are only "
            "excluded from reason codes. No fairness audit has been performed.",
            "Probabilities are not recalibrated.",
        ],
    }
