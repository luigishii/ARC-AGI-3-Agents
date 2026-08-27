import math
from agents.causal.perception import Scene, Object
from agents.causal.novelty import NoveltyModel, OPTIMISTIC_YIELD


def _scene(color=3, cell=(5, 5)):
    r, c = cell
    o = Object(color, frozenset([(r, c)]), (r, c, r, c), (float(r), float(c)), 1, "h")
    return Scene(objects=[o], grid=None)


def test_novelty_decreases_with_revisits():
    m = NoveltyModel()
    assert m.novelty("s") == 1.0                      # count 0 → 1/√1
    m.visit("s")
    assert math.isclose(m.novelty("s"), 1 / math.sqrt(2))
    m.visit("s")
    assert math.isclose(m.novelty("s"), 1 / math.sqrt(3))


def test_yield_estimate_optimistic_without_data():
    m = NoveltyModel()
    assert m.yield_estimate("ACTION6@cell=0,0") == OPTIMISTIC_YIELD


def test_observe_transition_updates_yield_and_counts():
    m = NoveltyModel()
    s = _scene()
    m.observe_transition("K", s)                      # 1º estado novo → novidade 1.0
    assert m.yield_estimate("K") == 1.0
    m.observe_transition("K", s)                       # revisita → novidade 1/√2
    assert math.isclose(m.yield_estimate("K"), (1.0 + 1 / math.sqrt(2)) / 2)


def test_record_goal_anchor_dedupes():
    m = NoveltyModel()
    m.record_goal_anchor("sigA")
    m.record_goal_anchor("sigA")
    m.record_goal_anchor("sigB")
    assert m.goal_anchors == ["sigA", "sigB"]


def test_roundtrip_serialization():
    m = NoveltyModel()
    m.observe_transition("K", _scene())
    m.record_goal_anchor("sigA")
    d = m.to_dict()
    m2 = NoveltyModel.from_dict(d)
    assert m2.to_dict() == d
    assert m2.yield_estimate("K") == m.yield_estimate("K")
    assert m2.goal_anchors == ["sigA"]
