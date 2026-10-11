import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.training.data_loader import load_data
from unittest.mock import patch, MagicMock
import pandas as pd

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "version" in response.json()

@patch('src.training.data_loader.os.path.exists', return_value=False)
@patch('src.training.data_loader.fetch_ucirepo')
def test_data_loader_mock(mock_fetch, _mock_exists, monkeypatch):
    monkeypatch.delenv('USE_SYNTHETIC_DATA', raising=False)
    # Mock return value
    mock_dataset = MagicMock()
    mock_dataset.data.features = pd.DataFrame({'pay_0': [1, 2], 'limit_bal': [1000, 2000]})
    mock_dataset.data.targets = pd.DataFrame({'default payment next month': [0, 1]})
    
    mock_fetch.return_value = mock_dataset
    
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'target' in df.columns # Renamed from default payment...
    assert 'pay_1' in df.columns # Renamed from pay_0
    assert df.shape == (2, 3) # 2 rows, 2 features + 1 target


def test_predict_returns_top5_contributions(monkeypatch):
    """Train a tiny pipeline on synthetic data and check /predict returns real contributions."""
    from sklearn.pipeline import Pipeline
    from src.api import main as api_main
    from src.api.schemas import PredictionRequest
    from src.training.mock_data import generate_synthetic_data
    from src.training.preprocess import get_preprocessor, split_features_target
    import xgboost as xgb

    df = generate_synthetic_data()
    X, y = split_features_target(df)
    cat = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    pipe = Pipeline([
        ('preprocessor', get_preprocessor(cat, num)),
        ('classifier', xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=0)),
    ]).fit(X, y)
    monkeypatch.setitem(api_main.models, 'pipeline', pipe)

    row = X.iloc[0].to_dict()
    payload = {k: row[k] for k in PredictionRequest.model_fields}
    resp = client.post('/predict', json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body['top_features']) == 5
    assert set(body['shap_values']) == set(body['top_features'])
