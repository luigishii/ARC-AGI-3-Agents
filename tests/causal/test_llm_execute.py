from dataclasses import replace
from agents.causal.perception import Scene, Object
from agents.causal.llm import execute_goal


def _obj(cells, color=3, oid=0):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return replace(Object(color, cset, bbox, centroid, len(cells), "h"), id=oid)


def _scene(objs):
    return Scene(objects=objs, grid=None)


def test_execute_press():
    assert execute_goal({"type": "press", "action": "ACTION1"}, _scene([]), {}) == "ACTION1"


def test_execute_click_cell():
    assert execute_goal({"type": "click_cell", "gx": 1, "gy": 2}, _scene([]), {}) == \
        "ACTION6@cell=1,2"


def test_execute_reach_moves_toward():
    scene = _scene([_obj([(0, 0)], color=3, oid=1), _obj([(0, 5)], color=7, oid=2)])
    goal = {"type": "reach", "avatar": {"id": 1}, "target": {"id": 2}}
    moves = {"ACTION1": (0, 1), "ACTION2": (0, -1)}
    assert execute_goal(goal, scene, moves) == "ACTION1"


def test_execute_reach_rarest_target():
    scene = _scene([
        _obj([(0, 0)], color=5, oid=1),   # avatar
        _obj([(0, 1)], color=3, oid=2),
        _obj([(0, 9)], color=9, oid=3),   # cor rara
        _obj([(9, 9)], color=3, oid=4),
    ])
    goal = {"type": "reach", "avatar": {"id": 1}, "target": "rarest"}
    moves = {"ACTION1": (0, 1), "ACTION2": (0, -1)}
    assert execute_goal(goal, scene, moves) == "ACTION1"


def test_execute_reach_none_without_moves_or_object():
    scene = _scene([_obj([(0, 0)], oid=1)])
    assert execute_goal({"type": "reach", "avatar": {"id": 1}, "target": {"id": 2}},
                        scene, {}) is None                       # sem moves
    assert execute_goal({"type": "reach", "avatar": {"id": 9}, "target": {"id": 1}},
                        scene, {"ACTION1": (0, 1)}) is None       # avatar inexistente
