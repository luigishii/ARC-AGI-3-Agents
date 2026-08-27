# agents/causal/novelty.py
from __future__ import annotations

import math

from .policy import cell_of

OPTIMISTIC_YIELD = 1.0


def state_signature(scene) -> str:
    parts = []
    for o in scene.objects:
        gx, gy = cell_of(int(round(o.centroid[1])), int(round(o.centroid[0])))
        parts.append((o.color, gx, gy))
    return ";".join(f"{c},{gx},{gy}" for (c, gx, gy) in sorted(parts))


class NoveltyModel:
    def __init__(self):
        self.counts = {}          # sig(str) -> int
        self._yield = {}          # action_key -> [soma, n]
        self.goal_anchors = []    # list[str]

    def count(self, sig) -> int:
        return self.counts.get(sig, 0)

    def novelty(self, sig) -> float:
        return 1.0 / math.sqrt(self.count(sig) + 1)

    def visit(self, sig) -> None:
        self.counts[sig] = self.counts.get(sig, 0) + 1

    def observe_transition(self, key, curr_scene) -> None:
        sig = state_signature(curr_scene)
        nov = self.novelty(sig)               # novidade ANTES de contar
        s, n = self._yield.get(key, [0.0, 0])
        self._yield[key] = [s + nov, n + 1]
        self.visit(sig)

    def yield_estimate(self, key) -> float:
        v = self._yield.get(key)
        if not v or v[1] == 0:
            return OPTIMISTIC_YIELD
        return v[0] / v[1]

    def record_goal_anchor(self, sig) -> None:
        if sig not in self.goal_anchors:
            self.goal_anchors.append(sig)

    def to_dict(self) -> dict:
        return {
            "counts": dict(self.counts),
            "yield": {k: list(v) for k, v in self._yield.items()},
            "goal_anchors": list(self.goal_anchors),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NoveltyModel":
        m = cls()
        m.counts = dict(d.get("counts", {}))
        m._yield = {k: list(v) for k, v in d.get("yield", {}).items()}
        m.goal_anchors = list(d.get("goal_anchors", []))
        return m
