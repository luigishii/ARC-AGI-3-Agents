from arcengine import GameAction
from agents.causal.perception import Scene, Object
from agents.causal.policy import candidates, cell_of, GRID_N, _object_cells


def _obj(cells, color=3):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return Object(color, cset, bbox, centroid, len(cells), "h")


def test_simple_action_one_candidate():
    scene = Scene(objects=[], grid=None)
    out = candidates(scene, [GameAction.ACTION1])
    assert len(out) == 1
    assert out[0].key == "ACTION1"
    assert out[0].x is None and out[0].y is None


def test_complex_action_emits_grid_candidates():
    scene = Scene(objects=[], grid=None)
    out = candidates(scene, [GameAction.ACTION6])
    assert len(out) == GRID_N * GRID_N
    keys = {c.key for c in out}
    assert len(keys) == GRID_N * GRID_N          # chaves distintas
    pts = {(c.x, c.y) for c in out}
    assert len(pts) == GRID_N * GRID_N           # pontos distintos
    for c in out:
        assert 0 <= c.x <= 63 and 0 <= c.y <= 63


def test_object_cells_marks_occupied_cell():
    # objeto em (row=5,col=5) → célula gx=0,gy=0
    scene = Scene(objects=[_obj([(5, 5)])], grid=None)
    occ = _object_cells(scene)
    assert occ == {(0, 0)}


def test_has_object_flag_only_on_occupied_cell():
    scene = Scene(objects=[_obj([(5, 5)])], grid=None)
    out = candidates(scene, [GameAction.ACTION6])
    occupied = [c for c in out if c.has_object]
    assert len(occupied) == 1
    assert cell_of(occupied[0].x, occupied[0].y) == (0, 0)


def test_empty_scene_no_has_object():
    scene = Scene(objects=[], grid=None)
    out = candidates(scene, [GameAction.ACTION6])
    assert all(c.has_object is False for c in out)
