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

@patch('src.training.data_loader.fetch_ucirepo')
def test_data_loader_mock(mock_fetch):
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
