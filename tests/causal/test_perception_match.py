import numpy as np
from agents.causal.perception import parse, match_objects


def _grid_with(pos, color=3, n=5):
    g = np.zeros((n, n), dtype=int)
    g[pos] = color
    return g.tolist()


def test_same_object_keeps_id_after_move():
    prev = match_objects(None, parse(_grid_with((1, 1))))
    id0 = prev.objects[0].id
    curr = match_objects(prev, parse(_grid_with((1, 2))))     # mesmo objeto, moveu 1 coluna
    assert curr.objects[0].id == id0


def test_new_object_gets_fresh_id():
    prev = match_objects(None, parse(_grid_with((1, 1))))
    g = np.zeros((5, 5), dtype=int)
    g[1, 1] = 3
    g[4, 4] = 7                                # objeto novo, cor diferente
    curr = match_objects(prev, parse(g.tolist()))
    ids = sorted(o.id for o in curr.objects)
    assert len(set(ids)) == 2                  # dois ids distintos
