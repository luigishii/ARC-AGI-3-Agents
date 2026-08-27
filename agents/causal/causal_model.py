# agents/causal/causal_model.py
from __future__ import annotations

from collections import namedtuple

Effect = namedtuple("Effect", "kind detail")


def compute_effect(prev, curr) -> Effect:
    prev_by_id = {o.id: o for o in prev.objects}
    curr_by_id = {o.id: o for o in curr.objects}
    changes = []
    for oid, po in prev_by_id.items():
        co = curr_by_id.get(oid)
        if co is None:
            changes.append(Effect("disappeared", oid))
        else:
            dr = round(co.centroid[0] - po.centroid[0])
            dc = round(co.centroid[1] - po.centroid[1])
            if (dr, dc) != (0, 0):
                changes.append(Effect("moved", (dr, dc)))
            elif co.color != po.color:
                changes.append(Effect("recolored", (po.color, co.color)))
            elif co.cells != po.cells:
                changes.append(Effect("morphed", (po.size, co.size)))
    for oid, co in curr_by_id.items():
        if oid not in prev_by_id:
            changes.append(Effect("appeared", oid))
    if not changes:
        return Effect("none", None)
    if len(changes) == 1:
        return changes[0]
    return Effect("structural", tuple(c.kind for c in changes))


def _effect_token(e: Effect) -> str:
    return f"{e.kind}:{e.detail}"


class CausalModel:
    def __init__(self):
        self.rules = {}                 # action_key -> {effect_token -> count}
        self.progress_keys = set()      # action_keys que levaram a level_up
        self._pred_hits = 0
        self._pred_total = 0

    def _bump(self, action_key, effect: Effect, level_up: bool = False):
        tok = _effect_token(effect)
        d = self.rules.setdefault(action_key, {})
        d[tok] = d.get(tok, 0) + 1
        if level_up:
            self.progress_keys.add(action_key)

    def observe(self, prev, action_key, curr, level_up=False) -> Effect:
        eff = compute_effect(prev, curr)
        self._bump(action_key, eff, level_up)
        return eff

    def predict(self, action_key):
        d = self.rules.get(action_key)
        if not d:
            return (None, 0.0)
        tok, cnt = max(d.items(), key=lambda kv: kv[1])
        total = sum(d.values())
        kind, detail = tok.split(":", 1)
        return (Effect(kind, detail), cnt / total)

    def is_progress(self, action_key) -> bool:
        return action_key in self.progress_keys

    def record_prediction(self, predicted, actual: Effect):
        if predicted is None:
            return
        self._pred_total += 1
        if predicted.kind == actual.kind:
            self._pred_hits += 1

    def stats(self) -> dict:
        stable = 0
        for d in self.rules.values():
            total = sum(d.values())
            if total and max(d.values()) / total >= 0.8:
                stable += 1
        return {
            "coverage_keys": len(self.rules),
            "n_rules": sum(len(d) for d in self.rules.values()),
            "stable_ratio": stable / len(self.rules) if self.rules else 0.0,
            "prediction_accuracy": self._pred_hits / self._pred_total if self._pred_total else 0.0,
        }

    def to_dict(self) -> dict:
        return {
            "rules": self.rules,
            "progress_keys": sorted(self.progress_keys),
            "pred_hits": self._pred_hits,
            "pred_total": self._pred_total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalModel":
        m = cls()
        m.rules = {k: dict(v) for k, v in d.get("rules", {}).items()}
        m.progress_keys = set(d.get("progress_keys", []))
        m._pred_hits = d.get("pred_hits", 0)
        m._pred_total = d.get("pred_total", 0)
        return m
