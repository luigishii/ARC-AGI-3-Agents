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
        else:
            for o in scene.objects:
                y = int(round(o.centroid[0]))    # row
                x = int(round(o.centroid[1]))    # col
                out.append(Candidate(action, x, y, action_key(action, o)))
    return out
