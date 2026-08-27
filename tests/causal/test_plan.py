from agents.causal.planning import plan, TransitionModel
from agents.causal.novelty import NoveltyModel


def test_plan_returns_none_without_data():
    m = TransitionModel()
    nov = NoveltyModel()
    assert plan("A", ["k1", "k2"], m, nov, []) is None


def test_plan_prefers_deeper_novel_state():
    m = TransitionModel()
    m.observe("A", "k1", "s1")
    m.observe("A", "k2", "s2")
    m.observe("s1", "k3", "s3")        # via k1 chega-se a s3 (2 passos)
    nov = NoveltyModel()
    for _ in range(50):
        nov.visit("s2")                # s2 muito visitado → baixa novidade
    assert plan("A", ["k1", "k2"], m, nov, [], depth=3) == "k1"


def test_plan_goal_directed_with_anchor():
    m = TransitionModel()
    m.observe("A", "k1", "3,0,0")      # k1 chega exatamente na âncora
    m.observe("A", "k2", "9,9,9")
    nov = NoveltyModel()
    assert plan("A", ["k1", "k2"], m, nov, ["3,0,0"], depth=2) == "k1"


def test_plan_frontier_is_attractive_no_anchor():
    m = TransitionModel()
    m.observe("A", "k1", "s1")         # k1 conhecido → estado s1
    nov = NoveltyModel()
    for _ in range(50):
        nov.visit("s1")                # s1 saturado (baixa novidade)
    assert plan("A", ["k1", "k2"], m, nov, [], depth=2) == "k2"
