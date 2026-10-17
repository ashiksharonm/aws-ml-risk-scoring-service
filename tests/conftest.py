import pytest
import xgboost as xgb
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.monitoring.drift import build_reference
from src.training.mock_data import generate_synthetic_data
from src.training.preprocess import get_preprocessor, split_features_target


@pytest.fixture(scope="session")
def synthetic():
    X, y = split_features_target(generate_synthetic_data())
    return X, y


@pytest.fixture(scope="session")
def pipelines(synthetic):
    """Tiny champion/challenger pair on synthetic data (mechanics only, not accuracy)."""
    X, y = synthetic
    cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    pre = get_preprocessor([], cols)
    champion = Pipeline(
        [
            ("preprocessor", clone(pre)),
            ("classifier", xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=0)),
        ]
    ).fit(X, y)
    challenger = Pipeline(
        [
            ("preprocessor", clone(pre)),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    ).fit(X, y)
    return champion, challenger


@pytest.fixture(scope="session")
def reference(synthetic, pipelines):
    X, _ = synthetic
    return build_reference(X, pipelines[0].predict_proba(X)[:, 1])
