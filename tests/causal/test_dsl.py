from dataclasses import replace
from agents.causal.perception import Scene, Object
from agents.causal import dsl


def _obj(cells, color=3, oid=0):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return replace(Object(color, cset, bbox, centroid, len(cells), "h"), id=oid)


def _scene(objs):
    return Scene(objects=objs, grid=None)


def test_objects_of_color_and_rarest():
    s = _scene([_obj([(0, 0)], color=3, oid=1), _obj([(0, 1)], color=3, oid=2),
                _obj([(0, 2)], color=9, oid=3)])
    assert len(dsl.objects_of_color(s, 3)) == 2
    assert dsl.rarest_color(s) == 9
    assert dsl.rarest_color(_scene([])) is None


def test_largest_smallest_nearest():
    a = _obj([(0, 0)], oid=1)                 # size 1
    b = _obj([(0, 1), (0, 2)], oid=2)         # size 2
    assert dsl.largest([a, b]) is b
    assert dsl.smallest([a, b]) is a
    assert dsl.nearest([a, b], (0, 3)) is b   # b mais perto de (0,3)


def test_accessors_and_spatial():
    o = _obj([(2, 4)], color=5, oid=7)
    assert dsl.ocolor(o) == 5 and dsl.osize(o) == 1 and dsl.oid(o) == 7
    assert dsl.ocentroid(o) == (2.0, 4.0)
    assert dsl.manhattan((0, 0), (1, 3)) == 4
    assert dsl.same_color(o, _obj([(9, 9)], color=5, oid=8)) is True


def test_action_builders():
    assert dsl.press("ACTION1") == "ACTION1"
    assert dsl.click(2, 3) == "ACTION6@cell=2,3"


def test_move_toward():
    av = _obj([(0, 0)], oid=1)
    tg = _obj([(0, 5)], oid=2)
    moves = {"ACTION1": (0, 1), "ACTION2": (0, -1)}
    assert dsl.move_toward(av, tg, moves) == "ACTION1"
    assert dsl.move_toward(av, tg, {}) is None


def test_DSL_dict_has_primitives():
    for name in ("objects_of_color", "rarest_color", "nearest", "ocentroid",
                 "manhattan", "press", "click", "move_toward"):
        assert name in dsl.DSL and callable(dsl.DSL[name])
