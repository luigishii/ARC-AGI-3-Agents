# agents/causal/policy.py
from __future__ import annotations

import random
from collections import namedtuple

from arcengine import GameAction

Candidate = namedtuple("Candidate", "action x y key has_object")

GRID_N = 6


def cell_center(gx: int, gy: int) -> tuple[int, int]:
    x = int((gx + 0.5) * 64 / GRID_N)
    y = int((gy + 0.5) * 64 / GRID_N)
    return x, y


def cell_of(x: int, y: int) -> tuple[int, int]:
    gx = min(GRID_N - 1, int(x) * GRID_N // 64)
    gy = min(GRID_N - 1, int(y) * GRID_N // 64)
    return gx, gy


def _as_action(a):
    # available_actions pode vir como int (id) ou GameAction. Normaliza p/ GameAction.
    return a if isinstance(a, GameAction) else GameAction.from_id(a)


def action_key(action, cell=None) -> str:
    if not action.is_complex():
        return action.name
    if cell is None:
        return f"{action.name}@empty"
    gx, gy = cell
    return f"{action.name}@cell={gx},{gy}"


def _object_cells(scene) -> set:
    occ = set()
    for o in scene.objects:
        for (r, c) in o.cells:
            occ.add(cell_of(c, r))   # x=col, y=row
    return occ


def candidates(scene, available_actions) -> list:
    out = []
    occ = _object_cells(scene)
    for a in available_actions:
        action = _as_action(a)
        if not action.is_complex():
            out.append(Candidate(action, None, None, action.name, False))
        else:
            for gy in range(GRID_N):
                for gx in range(GRID_N):
                    x, y = cell_center(gx, gy)
                    out.append(
                        Candidate(action, x, y, action_key(action, (gx, gy)),
                                  (gx, gy) in occ)
                    )
    return out


class Policy:
    def __init__(self, seed: int = 0, epsilon: float = 0.05):
        self._rng = random.Random(seed)
        self.epsilon = epsilon

    def score(self, cand, model, seen_effects, budget_frac) -> float:
        eff, conf = model.predict(cand.key)
        s = 0.0
        if model.is_progress(cand.key):
            s += 10.0 * (1 + (1 - budget_frac))
        if eff is None:
            s += 3.0
        elif conf < 0.8:
            s += 1.5
        if eff is not None and eff.kind not in seen_effects:
            s += 0.5
        if eff is not None and eff.kind == "none":
            s -= 2.0
        return s

    def decide(self, scene, model, available_actions, seen_effects, budget_frac):
        cands = candidates(scene, available_actions)
        if not cands:
            return None
        if self._rng.random() < self.epsilon:
            return self._rng.choice(cands)
        best, best_s = None, None
        for c in cands:
            sc = self.score(c, model, seen_effects, budget_frac)
            if best_s is None or sc > best_s:
                best, best_s = c, sc
        return best
