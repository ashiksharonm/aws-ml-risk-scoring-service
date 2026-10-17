"""Write two demo batches for the drift CLI by resampling the training data.

``stable.csv`` is a plain random sample; ``drifted.csv`` simulates a deteriorating
portfolio (more late payers, lower limits). Both are derived from the same dataset the
model was trained on, so the contrast is by construction, not a measured real-world shift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.data_loader import load_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/demo")
    parser.add_argument("--rows", type=int, default=500)
    args = parser.parse_args()

    df = load_data().drop(columns=["target"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stable = df.sample(args.rows, random_state=1)
    stable.to_csv(out / "stable.csv", index=False)

    drifted = df.sample(args.rows, random_state=2).copy()
    drifted["pay_1"] = (drifted["pay_1"] + 1).clip(upper=8)
    drifted["limit_bal"] = (drifted["limit_bal"] * 0.6).round()
    drifted.to_csv(out / "drifted.csv", index=False)
    print(f"wrote {out / 'stable.csv'} and {out / 'drifted.csv'} ({args.rows} rows each)")


if __name__ == "__main__":
    main()
