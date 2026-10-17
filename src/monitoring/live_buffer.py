"""Thread-safe rolling window of recent scoring requests for live drift checks."""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List, Tuple

import pandas as pd


class LiveBuffer:
    def __init__(self, maxlen: int) -> None:
        self._rows: Deque[Tuple[Dict[str, float], float]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, features: Dict[str, float], score: float) -> None:
        with self._lock:
            self._rows.append((features, score))

    def snapshot(self) -> Tuple[pd.DataFrame, List[float]]:
        with self._lock:
            rows = list(self._rows)
        return pd.DataFrame([r[0] for r in rows]), [r[1] for r in rows]

    def __len__(self) -> int:
        return len(self._rows)
