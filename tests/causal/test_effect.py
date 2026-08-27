import numpy as np
from agents.causal.perception import parse, match_objects
from agents.causal.causal_model import compute_effect


def _g(pos, color=3, n=6):
    g = np.zeros((n, n), dtype=int)
    g[pos] = color
    return g.tolist()


def _scene(prev, grid):
    return match_objects(prev, parse(grid))


def test_none_effect():
    s0 = _scene(None, _g((1, 1)))
    s1 = _scene(s0, _g((1, 1)))
    assert compute_effect(s0, s1).kind == "none"


def test_moved_effect():
    s0 = _scene(None, _g((1, 1)))
    s1 = _scene(s0, _g((1, 3)))
    e = compute_effect(s0, s1)
    assert e.kind == "moved" and e.detail == (0, 2)


def test_disappeared_effect():
    s0 = _scene(None, _g((1, 1)))
    s1 = _scene(s0, np.zeros((6, 6), dtype=int).tolist())
    assert compute_effect(s0, s1).kind == "disappeared"


def test_structural_effect_when_many_changes():
    g0 = np.zeros((6, 6), dtype=int); g0[0, 0] = 3; g0[5, 5] = 4
    g1 = np.zeros((6, 6), dtype=int); g1[0, 1] = 3; g1[4, 5] = 4
    s0 = _scene(None, g0.tolist())
    s1 = _scene(s0, g1.tolist())
    assert compute_effect(s0, s1).kind == "structural"
