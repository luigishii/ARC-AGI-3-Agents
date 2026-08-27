import numpy as np
from agents.causal.hud import HudMask


def _pair(changed_cells, shape=(8, 8)):
    a = np.zeros(shape, dtype=int)
    b = np.zeros(shape, dtype=int)
    for (r, c) in changed_cells:
        b[r, c] = 5
    return a, b


def test_mask_empty_before_min_samples():
    h = HudMask()
    for _ in range(4):                       # < HUD_MIN_SAMPLES (5)
        a, b = _pair([(0, 0)])
        h.update(a, b)
    assert not h.mask().any()                # cego ainda


def test_cell_changing_every_step_is_masked():
    h = HudMask()
    for _ in range(6):                       # >= 5 amostras, muda toda vez
        a, b = _pair([(0, 0)])
        h.update(a, b)
    m = h.mask()
    assert m[0, 0]                           # HUD
    assert not m[3, 3]                       # nunca mudou


def test_cell_changing_once_not_masked():
    h = HudMask()
    a, b = _pair([(1, 1)]); h.update(a, b)   # muda 1x
    for _ in range(9):                       # 9 transições sem mudanca
        z = np.zeros((8, 8), dtype=int)
        h.update(z, z)
    assert not h.mask()[1, 1]                # 1/10 = 0.1 < 0.7


def test_serialization_roundtrip():
    h = HudMask()
    for _ in range(6):
        a, b = _pair([(0, 1)]); h.update(a, b)
    h2 = HudMask.from_dict(h.to_dict())
    assert np.array_equal(h2.mask(), h.mask())
