from agents.causal.sandbox import execute_code_goal
from agents.causal.dsl import DSL

_SCENE = object()


def test_extra_injects_dsl():
    src = "def decide(scene):\n    return click(2, 3)\n"
    extra = {**DSL, "MOVES": {}}
    assert execute_code_goal(src, _SCENE, extra=extra) == "ACTION6@cell=2,3"


def test_extra_injects_moves():
    from dataclasses import replace
    from agents.causal.perception import Scene, Object

    def _obj(cells, color=3, oid=0):
        cset = frozenset(cells)
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        return replace(Object(color, cset, (0, 0, 0, 0),
                              (sum(rs) / len(rs), sum(cs) / len(cs)), 1, "h"), id=oid)

    scene = Scene(objects=[_obj([(0, 0)], oid=1), _obj([(0, 5)], color=9, oid=2)], grid=None)
    src = ("def decide(scene):\n"
           "    av = scene.objects[0]\n"
           "    tg = objects_of_color(scene, 9)[0]\n"
           "    return move_toward(av, tg, MOVES)\n")
    extra = {**DSL, "MOVES": {"ACTION1": (0, 1), "ACTION2": (0, -1)}}
    assert execute_code_goal(src, scene, extra=extra) == "ACTION1"


def test_no_extra_is_v10c_behavior():
    src = "def decide(scene):\n    return 'ACTION1'\n"
    assert execute_code_goal(src, _SCENE) == "ACTION1"
    # sem extra, click não está definido → NameError → None
    assert execute_code_goal("def decide(scene):\n    return click(1,1)\n", _SCENE) is None
