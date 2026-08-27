# agents/causal/policy.py
from __future__ import annotations

import random
from collections import namedtuple

from arcengine import GameAction

Candidate = namedtuple("Candidate", "action x y key")


def _as_action(a):
    # available_actions pode vir como int (id) ou GameAction. Normaliza p/ GameAction.
    return a if isinstance(a, GameAction) else GameAction.from_id(a)


def action_key(action, target_obj) -> str:
    if not action.is_complex():
        return action.name
    if target_obj is None:
        return f"{action.name}@empty"
    return f"{action.name}@color={target_obj.color},size={target_obj.size}"


def candidates(scene, available_actions) -> list:
    out = []
    for a in available_actions:
        action = _as_action(a)
        if not action.is_complex():
            out.append(Candidate(action, None, None, action.name))
        elif not scene.objects:
            # Cena sem objetos: emite 1 candidato de fallback no centro da
            # grade, senão a ação complexa fica sem nenhum candidato e
            # decide() poderia retornar None mesmo havendo ação disponível.
            out.append(Candidate(action, 32, 32, f"{action.name}@empty"))
        else:
            for o in scene.objects:
                y = int(round(o.centroid[0]))    # row
                x = int(round(o.centroid[1]))    # col
                out.append(Candidate(action, x, y, action_key(action, o)))
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
