import numpy as np
from arcengine import GameAction

from agents.causal.perception import parse, match_objects
from agents.causal.policy import candidates


def _scene():
    g = np.zeros((6, 6), dtype=int)
    g[1, 1] = 3
    g[4, 4] = 7
    return match_objects(None, parse(g.tolist()))


def test_simple_action_one_candidate():
    cands = candidates(_scene(), [GameAction.ACTION1])
    assert len(cands) == 1
    assert cands[0].x is None and cands[0].key == "ACTION1"


def test_complex_action_candidate_per_object():
    # ACTION6 é complexa (is_complex() == True); espera 1 candidato por objeto (2)
    cands = candidates(_scene(), [GameAction.ACTION6])
    assert len(cands) == 2
    coords = sorted((c.x, c.y) for c in cands)
    assert coords == [(1, 1), (4, 4)]        # (col,row) dos centroides
    assert all("@color=" in c.key for c in cands)
