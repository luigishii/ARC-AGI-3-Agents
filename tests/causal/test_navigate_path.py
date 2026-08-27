from dataclasses import replace
from agents.causal.perception import Scene, Object
from agents.causal.navigate import navigate, MovementModel


def _obj(cells, color=3, oid=0):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return replace(Object(color, cset, bbox, centroid, len(cells), "h"), id=oid)


def _move(vecs, avatar_id):
    m = MovementModel()
    for k, (dr, dc) in vecs.items():
        m.vec[k] = {(dr, dc): 1}
    m.avatar_counts[avatar_id] = 5
    return m


def test_navigate_none_without_data():
    scene = Scene(objects=[_obj([(0, 0)], oid=1)], grid=None)
    assert navigate(scene, MovementModel()) is None


def test_navigate_moves_toward_target():
    scene = Scene(objects=[_obj([(0, 0)], color=3, oid=1),
                           _obj([(0, 5)], color=7, oid=2)], grid=None)
    move = _move({"ACTION1": (0, 1), "ACTION2": (0, -1)}, avatar_id=1)
    assert navigate(scene, move) == "ACTION1"


def test_navigate_none_when_at_target():
    scene = Scene(objects=[_obj([(0, 0)], color=3, oid=1),
                           _obj([(0, 0)], color=7, oid=2)], grid=None)
    move = _move({"ACTION1": (0, 1)}, avatar_id=1)
    assert navigate(scene, move) is None


def test_navigate_targets_rarest_color():
    scene = Scene(objects=[
        _obj([(0, 0)], color=5, oid=1),
        _obj([(0, 1)], color=3, oid=2),
        _obj([(0, 9)], color=9, oid=3),
        _obj([(9, 9)], color=3, oid=4),
    ], grid=None)
    move = _move({"ACTION1": (0, 1), "ACTION2": (0, -1)}, avatar_id=1)
    assert navigate(scene, move) == "ACTION1"
