import numpy as np
from agents.causal.perception import parse, to_grid


def test_to_grid_reduces_stack():
    layer = np.zeros((3, 3), dtype=int)
    layer[0, 0] = 9
    g = to_grid([np.zeros((3, 3), dtype=int).tolist(), layer.tolist()])
    assert g[0, 0] == 9 and g.shape == (3, 3)


def test_hud_mask_excludes_masked_object():
    grid = np.zeros((6, 6), dtype=int)
    grid[0, 0] = 3            # objeto no HUD
    grid[4, 4] = 7            # objeto de gameplay
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True         # mascara o (0,0)
    scene = parse(grid.tolist(), hud_mask=mask)
    colors = sorted(o.color for o in scene.objects)
    assert colors == [7]      # o objeto mascarado sumiu da Scene


def test_no_mask_is_v1_behavior():
    grid = np.zeros((6, 6), dtype=int)
    grid[0, 0] = 3
    grid[4, 4] = 7
    scene = parse(grid.tolist())               # sem mask
    assert len(scene.objects) == 2
