"""Population Stability Index (PSI) drift monitoring against a training reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

EPS = 1e-4
N_BINS = 10
MODERATE = 0.10
SIGNIFICANT = 0.25
SCORE_KEY = "__score__"


def status_for(psi_value: float) -> str:
    if psi_value >= SIGNIFICANT:
        return "significant"
    if psi_value >= MODERATE:
        return "moderate"
    return "stable"


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, values, side="right")
    counts = np.bincount(bins, minlength=len(edges) + 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def psi(expected: Sequence[float] | np.ndarray, actual: Sequence[float] | np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _reference_entry(values: np.ndarray, n_bins: int) -> dict:
    values = values[~np.isnan(values)]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return {"edges": edges.tolist(), "expected": _proportions(values, edges).tolist()}


def build_reference(
    features: pd.DataFrame, scores: Optional[Iterable[float]] = None, n_bins: int = N_BINS
) -> dict:
    """Quantile-bin each numeric feature (and the score) of the training data."""
    ref: Dict[str, dict] = {}
    for col in features.select_dtypes(include=[np.number]).columns:
        ref[str(col)] = _reference_entry(features[col].to_numpy(dtype=float), n_bins)
    if scores is not None:
        ref[SCORE_KEY] = _reference_entry(np.asarray(list(scores), dtype=float), n_bins)
    return {"n_rows": int(len(features)), "n_bins": n_bins, "features": ref}


def save_reference(reference: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(reference))


def load_reference(path: str) -> dict:
    return json.loads(Path(path).read_text())


def drift_report(
    batch: pd.DataFrame,
    reference: dict,
    scores: Optional[Iterable[float]] = None,
    min_rows: int = 100,
) -> dict:
    """PSI per feature plus score-distribution drift for ``batch`` vs ``reference``."""
    results: List[dict] = []

    def add(name: str, values: np.ndarray, entry: dict) -> None:
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return
        edges = np.asarray(entry["edges"], dtype=float)
        value = psi(entry["expected"], _proportions(values, edges))
        results.append({"feature": name, "psi": round(value, 4), "status": status_for(value)})

    for name, entry in reference["features"].items():
        if name == SCORE_KEY or name not in batch.columns:
            continue
        add(name, pd.to_numeric(batch[name], errors="coerce").to_numpy(dtype=float), entry)

    score_result = None
    if scores is not None and SCORE_KEY in reference["features"]:
        add(SCORE_KEY, np.asarray(list(scores), dtype=float), reference["features"][SCORE_KEY])
        score_result = next((r for r in results if r["feature"] == SCORE_KEY), None)
        results = [r for r in results if r["feature"] != SCORE_KEY]
    results.sort(key=lambda r: r["psi"], reverse=True)

    order = {"stable": 0, "moderate": 1, "significant": 2}
    overall = max(
        [r["status"] for r in results] + ([score_result["status"]] if score_result else []),
        key=lambda status: order[status],
        default="stable",
    )
    return {
        "n_rows": int(len(batch)),
        "reference_rows": reference.get("n_rows"),
        "overall_status": overall,
        "score_drift": score_result,
        "drifted_features": [r["feature"] for r in results if r["status"] != "stable"],
        "features": results,
        "warnings": (
            [f"batch has {len(batch)} rows (< {min_rows}); PSI is noisy on small samples"]
            if len(batch) < min_rows
            else []
        ),
        "thresholds": {"moderate": MODERATE, "significant": SIGNIFICANT},
    }
