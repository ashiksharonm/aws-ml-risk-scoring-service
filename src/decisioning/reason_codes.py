"""Adverse-action reason codes derived from SHAP contributions.

For a declined applicant, regulators (e.g. ECOA / Reg B in the US) expect the principal
reasons for the decision. We take the features that push the default probability *up*
the most (positive log-odds SHAP), map them to human-readable reason codes, merge
features that share a reason, and never surface protected or proxy attributes as a reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Reason:
    code: str
    text: str


def _build_map() -> Dict[str, Reason]:
    mapping: Dict[str, Reason] = {
        "pay_1": Reason("R01", "Delinquent or revolving repayment status in the most recent month"),
        "limit_bal": Reason("R03", "Low assigned credit limit"),
    }
    for i in range(2, 7):
        mapping[f"pay_{i}"] = Reason("R02", "Delinquent repayment history in prior months")
    for i in range(1, 7):
        mapping[f"bill_amt{i}"] = Reason("R04", "High outstanding statement balances")
        mapping[f"pay_amt{i}"] = Reason("R05", "Low recent payment amounts")
    return mapping


FEATURE_REASONS: Dict[str, Reason] = _build_map()

# Never reported as a reason for adverse action (protected class or likely proxy).
EXCLUDED_FEATURES = frozenset({"sex", "age", "marriage", "education"})


def base_feature(transformed_name: str) -> str:
    """'num__pay_1' -> 'pay_1' (strip the ColumnTransformer prefix)."""
    return transformed_name.split("__", 1)[-1]


def adverse_action_reasons(
    feature_names: Sequence[str],
    contributions: Sequence[float],
    top_n: int = 4,
    applicant: Mapping[str, float] | None = None,
) -> List[dict]:
    """Top-``top_n`` reasons, strongest first. Returns [] when nothing raises risk."""
    grouped: Dict[str, dict] = {}
    for name, value in zip(feature_names, np.asarray(contributions, dtype=float)):
        feature = base_feature(name)
        if feature in EXCLUDED_FEATURES or value <= 0 or feature not in FEATURE_REASONS:
            continue
        reason = FEATURE_REASONS[feature]
        entry = grouped.setdefault(
            reason.code,
            {"code": reason.code, "reason": reason.text, "contribution": 0.0, "features": []},
        )
        entry["contribution"] += float(value)
        entry["features"].append(feature)
    ranked = sorted(grouped.values(), key=lambda e: e["contribution"], reverse=True)[:top_n]
    for entry in ranked:
        entry["contribution"] = round(entry["contribution"], 4)
        if applicant is not None:
            entry["applicant_values"] = {f: applicant[f] for f in entry["features"] if f in applicant}
    return ranked
