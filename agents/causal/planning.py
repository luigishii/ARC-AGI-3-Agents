# agents/causal/planning.py
from __future__ import annotations

PLAN_DEPTH = 3
PLAN_BEAM = 8


class TransitionModel:
    def __init__(self):
        self.trans = {}    # sig -> {key -> {next_sig -> count}}

    def observe(self, prev_sig, key, next_sig) -> None:
        d = self.trans.setdefault(prev_sig, {}).setdefault(key, {})
        d[next_sig] = d.get(next_sig, 0) + 1

    def predict_next(self, sig, key):
        d = self.trans.get(sig, {}).get(key)
        if not d:
            return None
        return max(d.items(), key=lambda kv: kv[1])[0]

    def known_keys(self, sig):
        return list(self.trans.get(sig, {}).keys())

    def to_dict(self) -> dict:
        return {s: {k: dict(nn) for k, nn in kk.items()}
                for s, kk in self.trans.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionModel":
        m = cls()
        m.trans = {s: {k: dict(nn) for k, nn in kk.items()}
                   for s, kk in d.items()}
        return m
