# agents/causal/transfer.py
from __future__ import annotations

import threading

W_PRIOR = 1.0
NEUTRAL_PRODUCTIVITY = 0.5


def abstract_feature(cand) -> str:
    if not cand.action.is_complex():
        return "simple"
    return "click_on_object" if cand.has_object else "click_empty"


class TransferPrior:
    def __init__(self):
        self._counts = {}          # feature -> [n_produtivo, n_total]
        self._lock = threading.Lock()

    def observe(self, feature, effect_kind) -> None:
        with self._lock:
            c = self._counts.setdefault(feature, [0, 0])
            c[1] += 1
            if effect_kind not in (None, "none"):
                c[0] += 1

    def productivity(self, feature) -> float:
        with self._lock:
            c = self._counts.get(feature)
            if not c or c[1] == 0:
                return NEUTRAL_PRODUCTIVITY
            return c[0] / c[1]

    def to_dict(self) -> dict:
        with self._lock:
            return {"counts": {k: list(v) for k, v in self._counts.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "TransferPrior":
        p = cls()
        p._counts = {k: list(v) for k, v in d.get("counts", {}).items()}
        return p


_SHARED = TransferPrior()


def shared_prior() -> TransferPrior:
    return _SHARED


def reset_shared_prior() -> None:
    global _SHARED
    _SHARED = TransferPrior()
