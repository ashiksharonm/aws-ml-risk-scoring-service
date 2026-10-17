import json

from src.training import train as train_mod
from src.utils.config import settings


def test_train_writes_all_artifacts(tmp_path, monkeypatch):
    """Runs the real training entrypoint on the 1000-row synthetic set (seconds on CPU)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_SYNTHETIC_DATA", "true")
    for name, fname in [
        ("MODEL_PATH", "model.pkl"),
        ("CHALLENGER_PATH", "challenger.pkl"),
        ("REFERENCE_PATH", "reference.json"),
        ("MODEL_CARD_PATH", "model_card.json"),
    ]:
        monkeypatch.setattr(settings, name, str(tmp_path / "models" / fname))
    train_mod.train()
    for fname in ["model.pkl", "challenger.pkl", "reference.json", "model_card.json"]:
        assert (tmp_path / "models" / fname).exists()
    card = json.loads((tmp_path / "models" / "model_card.json").read_text())
    assert card["training_data"]["source"].startswith("synthetic")
    assert "pay_1" in json.loads((tmp_path / "models" / "reference.json").read_text())["features"]
