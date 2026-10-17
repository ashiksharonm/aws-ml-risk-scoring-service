"""Champion/challenger shadow scoring.

The challenger scores every request but its output is only logged; it never changes the
decision returned to the caller.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ShadowScorer:
    def __init__(self, challenger: Any, log_path: str, threshold: float) -> None:
        self.challenger = challenger
        self.log_path = Path(log_path)
        self.threshold = threshold
        self._lock = threading.Lock()
        self.n = 0
        self.agree = 0
        self.abs_diff_sum = 0.0

    def score(
        self,
        frame: pd.DataFrame,
        champion_prob: float,
        request_id: str,
        threshold: Optional[float] = None,
    ) -> Optional[dict]:
        """Score with the challenger and log it. Never raises."""
        thr = self.threshold if threshold is None else threshold
        try:
            challenger_prob = float(self.challenger.predict_proba(frame)[:, 1][0])
            record = {
                "request_id": request_id,
                "champion_probability": round(champion_prob, 6),
                "challenger_probability": round(challenger_prob, 6),
                "champion_decision": int(champion_prob >= thr),
                "challenger_decision": int(challenger_prob >= thr),
            }
            record["agree"] = record["champion_decision"] == record["challenger_decision"]
            with self._lock:
                self.n += 1
                self.agree += 1 if record["agree"] else 0
                self.abs_diff_sum += abs(champion_prob - challenger_prob)
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
            return record
        except Exception:
            logger.exception("shadow scoring failed")
            return None

    def summary(self) -> dict:
        with self._lock:
            n = self.n
            return {
                "scored": n,
                "decision_agreement": (self.agree / n) if n else None,
                "mean_abs_probability_diff": (self.abs_diff_sum / n) if n else None,
                "log_path": str(self.log_path),
            }
