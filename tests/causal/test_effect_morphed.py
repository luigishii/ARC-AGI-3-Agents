import numpy as np
from agents.causal.perception import parse, match_objects
from agents.causal.causal_model import compute_effect


def _scene(prev, grid):
    return match_objects(prev, parse(grid))


def test_morphed_on_inplace_resize():
    # objeto em cruz cujo centroide NAO muda ao ganhar celulas simetricas
    g0 = np.zeros((9, 9), dtype=int)
    g0[4, 4] = 3; g0[4, 3] = 3; g0[4, 5] = 3          # linha horizontal, centroide (4,4)
    s0 = _scene(None, g0.tolist())
    g1 = np.zeros((9, 9), dtype=int)
    g1[4, 4] = 3; g1[4, 3] = 3; g1[4, 5] = 3; g1[3, 4] = 3; g1[5, 4] = 3  # cruz, centroide (4,4)
    s1 = _scene(s0, g1.tolist())
    e = compute_effect(s0, s1)
    assert e.kind == "morphed"                        # id mantido (IoU 3/5), centroide igual, cells mudou


def test_moved_still_moved():
    g0 = np.zeros((8, 8), dtype=int); g0[1, 1] = 3
    s0 = _scene(None, g0.tolist())
    g1 = np.zeros((8, 8), dtype=int); g1[1, 4] = 3
    s1 = _scene(s0, g1.tolist())
    assert compute_effect(s0, s1).kind == "moved"
