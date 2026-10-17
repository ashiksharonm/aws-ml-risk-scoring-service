import numpy as np
import pandas as pd
import pytest

from src.monitoring import cli
from src.monitoring.drift import (
    build_reference,
    drift_report,
    load_reference,
    psi,
    save_reference,
    status_for,
)


def test_psi_identical_is_zero_and_shift_is_positive():
    p = [0.25, 0.25, 0.25, 0.25]
    assert psi(p, p) == pytest.approx(0.0)
    assert psi(p, [0.7, 0.1, 0.1, 0.1]) > 0.25


def test_status_thresholds():
    assert status_for(0.05) == "stable"
    assert status_for(0.1) == "moderate"
    assert status_for(0.25) == "significant"


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=5000), "b": rng.integers(0, 5, 5000)})


def test_same_distribution_is_stable(frame):
    ref = build_reference(frame, np.random.default_rng(1).random(len(frame)))
    sample = frame.sample(1000, random_state=3)
    scores = np.random.default_rng(2).random(1000)
    rep = drift_report(sample, ref, scores=scores)
    assert rep["overall_status"] == "stable"
    assert rep["score_drift"]["status"] == "stable"
    assert rep["warnings"] == []


def test_shifted_feature_and_scores_are_flagged(frame):
    ref = build_reference(frame, np.random.default_rng(1).random(len(frame)))
    shifted = frame.sample(1000, random_state=3).assign(a=lambda d: d["a"] + 2)
    rep = drift_report(shifted, ref, scores=np.full(1000, 0.9))
    assert rep["drifted_features"] == ["a"]
    assert rep["features"][0]["feature"] == "a"
    assert rep["score_drift"]["status"] == "significant"
    assert rep["overall_status"] == "significant"


def test_small_batch_warns_and_reference_roundtrips(frame, tmp_path):
    ref = build_reference(frame)
    path = str(tmp_path / "ref.json")
    save_reference(ref, path)
    assert load_reference(path) == ref
    rep = drift_report(frame.head(10), ref)
    assert rep["warnings"] and rep["score_drift"] is None


def test_cli_end_to_end(frame, tmp_path, capsys):
    ref_path = str(tmp_path / "ref.json")
    save_reference(build_reference(frame), ref_path)
    ok = tmp_path / "ok.csv"
    frame.sample(500, random_state=1).to_csv(ok, index=False)
    bad = tmp_path / "bad.csv"
    frame.sample(500, random_state=1).assign(a=lambda d: d["a"] + 3).to_csv(bad, index=False)
    out = tmp_path / "report.json"

    args = ["drift", str(ok), "--reference", ref_path, "--model", "missing.pkl"]
    assert cli.main(args + ["--fail-on", "significant"]) == 0
    code = cli.main(
        [
            "drift",
            str(bad),
            "--reference",
            ref_path,
            "--model",
            "missing.pkl",
            "--fail-on",
            "significant",
            "--output",
            str(out),
        ]
    )
    assert code == 2
    assert "SIGNIFICANT" in capsys.readouterr().out
    assert out.exists()


def test_cli_scores_batch_with_model(pipelines, synthetic, reference, tmp_path):
    import joblib

    X, _ = synthetic
    model = tmp_path / "m.pkl"
    joblib.dump(pipelines[0], model)
    ref = tmp_path / "ref.json"
    save_reference(reference, str(ref))
    csv = tmp_path / "b.csv"
    X.sample(300, random_state=0).rename(columns={"pay_1": "pay_0"}).to_csv(csv, index=False)
    out = tmp_path / "r.json"
    assert (
        cli.main(["drift", str(csv), "--reference", str(ref), "--model", str(model), "--output", str(out)])
        == 0
    )
    assert '"score_drift": {' in out.read_text()
