from dataclasses import replace
from agents.causal.perception import Scene, Object
from agents.causal.navigate import _moved_object, MovementModel


def _obj(cells, color=3, oid=0):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    o = Object(color, cset, bbox, centroid, len(cells), "h")
    return replace(o, id=oid)


def _scene(objs):
    return Scene(objects=objs, grid=None)


def test_moved_object_single():
    prev = _scene([_obj([(0, 0)], oid=1)])
    curr = _scene([_obj([(0, 1)], oid=1)])
    assert _moved_object(prev, curr) == (1, (0, 1))


def test_moved_object_ambiguous_is_none():
    # 2 movers rigidos com vetores DISTINTOS = 2 entidades -> ambiguo (mesmo vetor = composto)
    prev = _scene([_obj([(0, 0)], oid=1), _obj([(5, 5)], oid=2)])
    curr = _scene([_obj([(0, 1)], oid=1), _obj([(6, 5)], oid=2)])
    assert _moved_object(prev, curr) is None


def test_moved_object_none_when_static():
    prev = _scene([_obj([(0, 0)], oid=1)])
    curr = _scene([_obj([(0, 0)], oid=1)])
    assert _moved_object(prev, curr) is None


def test_movement_model_learns_vector_and_avatar():
    m = MovementModel()
    m.observe("ACTION1", _scene([_obj([(0, 0)], oid=1)]),
              _scene([_obj([(0, 1)], oid=1)]))
    m.observe("ACTION1", _scene([_obj([(0, 1)], oid=1)]),
              _scene([_obj([(0, 2)], oid=1)]))
    assert m.move_vector("ACTION1") == (0, 1)
    assert m.avatar_id() == 1


def test_moves_excludes_complex_keys():
    m = MovementModel()
    m.observe("ACTION1", _scene([_obj([(0, 0)], oid=1)]),
              _scene([_obj([(0, 1)], oid=1)]))
    m.vec["ACTION6@cell=0,0"] = {(0, 1): 3}
    assert set(m.moves().keys()) == {"ACTION1"}


def test_roundtrip_serialization():
    m = MovementModel()
    m.observe("ACTION1", _scene([_obj([(0, 0)], oid=1)]),
              _scene([_obj([(0, 1)], oid=1)]))
    m2 = MovementModel.from_dict(m.to_dict())
    assert m2.to_dict() == m.to_dict()
    assert m2.move_vector("ACTION1") == (0, 1)
