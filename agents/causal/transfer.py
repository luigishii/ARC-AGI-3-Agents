# agents/causal/transfer.py
from __future__ import annotations

import json
import os
import threading

W_PRIOR = 1.0
NEUTRAL_PRODUCTIVITY = 0.5
DEFAULT_PRIOR_PATH = "agents/causal/prior.json"


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

    def merge(self, other) -> None:
        for feat, (np_, nt) in other.to_dict()["counts"].items():
            with self._lock:
                c = self._counts.setdefault(feat, [0, 0])
                c[0] += np_
                c[1] += nt


def save_prior(prior, path) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prior.to_dict(), f)
    os.replace(tmp, path)


def load_prior(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return TransferPrior.from_dict(json.load(f))


_SHARED = TransferPrior()
_load_lock = threading.Lock()
_loaded = False


def shared_prior() -> TransferPrior:
    return _SHARED


def reset_shared_prior() -> None:
    global _SHARED, _loaded
    _SHARED = TransferPrior()
    _loaded = False


def load_shared_once(path) -> None:
    global _loaded
    with _load_lock:
        if _loaded:
            return
        _loaded = True
        disk = load_prior(path)
        if disk is not None:
            shared_prior().merge(disk)
