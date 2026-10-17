"""Drift report CLI.

python -m src.monitoring.cli drift batch.csv
python -m src.monitoring.cli drift batch.csv --output report.json --fail-on significant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.monitoring.drift import drift_report, load_reference
from src.utils.config import settings

_RENAMES = {"pay_0": "pay_1"}
_ORDER = {"stable": 0, "moderate": 1, "significant": 2}


def load_batch(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame.rename(columns=_RENAMES)


def _score(batch: pd.DataFrame, model_path: str) -> Optional[List[float]]:
    if "score" in batch.columns:
        return batch["score"].astype(float).tolist()
    if not Path(model_path).exists():
        return None
    import joblib  # imported lazily: only needed when scoring the batch

    pipeline = joblib.load(model_path)
    cols = list(pipeline.feature_names_in_)
    return pipeline.predict_proba(batch[cols])[:, 1].tolist()


def format_report(report: dict) -> str:
    lines = [
        f"Drift report: {report['n_rows']} rows vs reference of {report['reference_rows']} rows",
        f"Overall status: {report['overall_status'].upper()}",
    ]
    if report["score_drift"]:
        sd = report["score_drift"]
        lines.append(f"Score distribution PSI: {sd['psi']:.4f} ({sd['status']})")
    lines.append(f"{'feature':<14}{'PSI':>8}  status")
    for r in report["features"]:
        lines.append(f"{r['feature']:<14}{r['psi']:>8.4f}  {r['status']}")
    lines += [f"warning: {w}" for w in report["warnings"]]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="risk-monitor")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("drift", help="PSI feature drift + score drift for a CSV batch")
    d.add_argument("csv", help="batch of applications (column names as in the API schema)")
    d.add_argument("--reference", default=settings.REFERENCE_PATH)
    d.add_argument("--model", default=settings.MODEL_PATH, help="used to score the batch")
    d.add_argument("--output", help="write the JSON report here")
    d.add_argument("--fail-on", choices=["moderate", "significant"], help="non-zero exit at this level")
    args = parser.parse_args(argv)

    reference = load_reference(args.reference)
    batch = load_batch(args.csv)
    scores = _score(batch, args.model)
    report = drift_report(batch, reference, scores=scores, min_rows=settings.MIN_DRIFT_ROWS)
    print(format_report(report))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    if args.fail_on and _ORDER[report["overall_status"]] >= _ORDER[args.fail_on]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
