import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.main import app
from src.api.schemas import PredictionRequest
from src.decisioning.shadow import ShadowScorer
from src.training.data_loader import load_data
from src.utils.config import settings

client = TestClient(app)


def test_health_check():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "version" in response.json()
    assert client.get("/health").status_code == 200


def test_readyz_503_without_model():
    api_main.models.clear()
    assert client.get("/readyz").status_code == 503
    assert client.post("/predict", json={}).status_code == 422


@patch("src.training.data_loader.os.path.exists", return_value=False)
@patch("src.training.data_loader.fetch_ucirepo")
def test_data_loader_mock(mock_fetch, _mock_exists, monkeypatch):
    monkeypatch.delenv("USE_SYNTHETIC_DATA", raising=False)
    mock_dataset = MagicMock()
    mock_dataset.data.features = pd.DataFrame({"pay_0": [1, 2], "limit_bal": [1000, 2000]})
    mock_dataset.data.targets = pd.DataFrame({"default payment next month": [0, 1]})
    mock_fetch.return_value = mock_dataset

    df = load_data()
    assert not df.empty
    assert "target" in df.columns
    assert "pay_1" in df.columns
    assert df.shape == (2, 3)


@pytest.fixture
def served(monkeypatch, pipelines, reference, tmp_path):
    champion, challenger = pipelines
    api_main.models.clear()
    api_main.models["pipeline"] = champion
    api_main.models["reference"] = reference
    api_main.models["shadow"] = ShadowScorer(challenger, str(tmp_path / "shadow.jsonl"), 0.5)
    return tmp_path / "shadow.jsonl"


def _payload(X, i=0):
    row = X.iloc[i].to_dict()
    return {k: row[k] for k in PredictionRequest.model_fields}


def test_predict_returns_top5_contributions(served, synthetic):
    X, _ = synthetic
    resp = client.post("/predict", json=_payload(X))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["top_features"]) == 5
    assert set(body["shap_values"]) == set(body["top_features"])
    assert resp.headers["x-request-id"] == body["request_id"]


def test_decline_has_reasons_and_approve_does_not(served, synthetic, monkeypatch):
    X, _ = synthetic
    monkeypatch.setattr(settings, "DECISION_THRESHOLD", 0.0)
    declined = client.post("/predict", json=_payload(X)).json()
    assert declined["decision"] == "decline" and declined["is_default"] == 1
    protected = {"sex", "age", "marriage", "education"}
    for r in declined["adverse_action_reasons"]:
        assert r["contribution"] > 0 and not protected & set(r["features"])
    monkeypatch.setattr(settings, "DECISION_THRESHOLD", 1.1)
    approved = client.post("/predict", json=_payload(X)).json()
    assert approved["decision"] == "approve" and approved["adverse_action_reasons"] == []


def test_request_id_is_propagated(served, synthetic):
    X, _ = synthetic
    resp = client.post("/predict", json=_payload(X), headers={"X-Request-ID": "abc123"})
    assert resp.json()["request_id"] == "abc123"


def test_shadow_logs_without_changing_decision(served, synthetic):
    X, _ = synthetic
    with_shadow = client.post("/predict", json=_payload(X)).json()
    api_main.models.pop("shadow")
    without = client.post("/predict", json=_payload(X)).json()
    assert with_shadow["default_probability"] == without["default_probability"]
    assert with_shadow["decision"] == without["decision"]
    lines = served.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert {"champion_probability", "challenger_probability", "agree"} <= set(rec)


def test_shadow_failure_never_raises(tmp_path):
    bad = MagicMock()
    bad.predict_proba.side_effect = RuntimeError("boom")
    scorer = ShadowScorer(bad, str(tmp_path / "s.jsonl"), 0.5)
    assert scorer.score(pd.DataFrame([{"a": 1}]), 0.3, "rid") is None


def test_shadow_endpoint(served, synthetic):
    X, _ = synthetic
    client.post("/predict", json=_payload(X))
    body = client.get("/monitoring/shadow").json()
    assert body["scored"] == 1
    api_main.models.pop("shadow")
    assert client.get("/monitoring/shadow").status_code == 404


def test_drift_endpoint_stable_vs_shifted(served, synthetic):
    X, _ = synthetic
    stable = [_payload(X, i) for i in range(300)]
    ok = client.post("/monitoring/drift", json={"rows": stable}).json()
    assert ok["overall_status"] == "stable"
    shifted = [dict(r, pay_1=8, limit_bal=r["limit_bal"] * 0.1) for r in stable]
    bad = client.post("/monitoring/drift", json={"rows": shifted}).json()
    assert bad["overall_status"] == "significant"
    assert "pay_1" in bad["drifted_features"]


def test_live_drift_and_model_card(served, synthetic):
    X, _ = synthetic
    api_main.live_buffer._rows.clear()
    assert client.get("/monitoring/live").status_code == 404
    for i in range(5):
        client.post("/predict", json=_payload(X, i))
    live = client.get("/monitoring/live").json()
    assert live["n_rows"] == 5 and live["warnings"]
    assert client.get("/model-card").status_code == 404
    api_main.models["model_card"] = {"model": "x"}
    assert client.get("/model-card").json() == {"model": "x"}


def test_demo_page_served():
    resp = client.get("/")
    assert resp.status_code == 200 and "/predict" in resp.text
