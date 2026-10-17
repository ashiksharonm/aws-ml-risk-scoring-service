"""Per-feature TreeSHAP contributions via XGBoost's native ``pred_contribs``."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


def compute_contributions(pipeline: Any, frame: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Return (feature_names, contributions) for each row of ``frame``.

    Contributions are log-odds of the positive class (default); positive values push
    towards default. The bias column is dropped. Shape: (n_rows, n_features).
    """
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(frame)
    names = [str(n) for n in preprocessor.get_feature_names_out()]
    matrix = xgb.DMatrix(transformed, feature_names=names)
    contribs = model.get_booster().predict(matrix, pred_contribs=True)
    return names, np.asarray(contribs)[:, :-1]


def top_contributions(names: List[str], row: np.ndarray, n: int = 5) -> dict:
    order = np.argsort(-np.abs(row))[:n]
    return {names[k]: float(row[k]) for k in order}
