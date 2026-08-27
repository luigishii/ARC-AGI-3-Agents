from dataclasses import replace
from agents.causal.perception import Scene, Object
from agents.causal.llm import NullLLMClient, build_prompt, parse_goal


class FakeLLM:
    def __init__(self, canned):
        self.canned = canned

    def complete(self, prompt):
        return self.canned


def _obj(cells, color=3, oid=0):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return replace(Object(color, cset, bbox, centroid, len(cells), "h"), id=oid)


def _scene():
    return Scene(objects=[_obj([(0, 0)], color=3, oid=1),
                          _obj([(0, 5)], color=7, oid=2)], grid=None)


def test_null_and_fake_client():
    assert NullLLMClient().complete("x") == ""
    assert FakeLLM("hi").complete("x") == "hi"


def test_build_prompt_contains_state_and_instruction():
    p = build_prompt(_scene(), {"available": ["ACTION1"], "moves": {"ACTION1": (0, 1)}})
    assert "OBJECTS" in p
    assert "color=3" in p and "color=7" in p
    assert "AVAILABLE_ACTIONS" in p
    assert "moves" in p
    assert "JSON" in p or "type" in p
    assert build_prompt(_scene(), {}) == build_prompt(_scene(), {})   # determinístico


def test_parse_goal_press():
    assert parse_goal('sure: {"type":"press","action":"ACTION1"} done') == \
        {"type": "press", "action": "ACTION1"}


def test_parse_goal_click_and_reach():
    assert parse_goal('{"type":"click_cell","gx":1,"gy":2}') == \
        {"type": "click_cell", "gx": 1, "gy": 2}
    g = parse_goal('{"type":"reach","avatar":{"id":1},"target":"rarest"}')
    assert g["type"] == "reach" and g["target"] == "rarest"


def test_parse_goal_rejects_bad():
    assert parse_goal("no json here") is None
    assert parse_goal("{not valid") is None
    assert parse_goal('{"type":"unknown"}') is None
    assert parse_goal('{"type":"press"}') is None      # falta action
