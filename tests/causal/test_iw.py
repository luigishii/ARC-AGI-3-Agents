from agents.causal.iw import atoms, iw_search, iw_plan
from agents.causal.typed_model import TypedWorldModel

_MOVE = "def transition(obj, action, ctx):\n    o = dict(obj)\n    o['x'] = o['x'] + 1\n    return o\n"


def _model():
    m = TypedWorldModel()
    m.set_rule("t", _MOVE)
    return m


def _start(x=0):
    return [("t", {"x": x, "y": 0, "color": 3})]


# --- atoms ---
def test_atoms_are_indexed_attribute_values():
    st = [("t", {"x": 1, "y": 2, "color": 3})]
    ats = atoms(st)
    assert (0, "x", 1) in ats
    assert (0, "y", 2) in ats
    assert (0, "color", 3) in ats


# --- iw_search rumo a um goal ---
def test_iw_reaches_goal():
    goal = lambda st: st[0][1]["x"] >= 2
    assert iw_search(_start(0), ["A"], _model(), goal_fn=goal, width=1) == "A"


def test_iw_none_when_goal_unreachable_in_budget():
    goal = lambda st: st[0][1]["x"] >= 999
    assert iw_search(_start(0), ["A"], _model(), goal_fn=goal, width=1, max_nodes=10) is None


def test_iw_none_when_no_actions():
    goal = lambda st: st[0][1]["x"] >= 2
    assert iw_search(_start(0), [], _model(), goal_fn=goal, width=1) is None


# --- modo exploração (sem goal): ação que gera novidade ---
def test_iw_exploration_returns_novel_action():
    assert iw_search(_start(0), ["A"], _model(), goal_fn=None, width=1) == "A"


def test_iw_exploration_none_when_no_change():
    noop = TypedWorldModel()
    noop.set_rule("t", "def transition(obj, action, ctx):\n    return dict(obj)\n")
    assert iw_search(_start(0), ["A"], noop, goal_fn=None, width=1) is None


# --- iw_plan escala a largura ---
def test_iw_plan_finds_goal():
    goal = lambda st: st[0][1]["x"] >= 3
    assert iw_plan(_start(0), ["A"], _model(), goal_fn=goal, max_width=2) == "A"


def test_iw_plan_pruning_terminates():
    # largura 1 poda estados sem átomo novo → busca sempre termina mesmo com goal impossível
    goal = lambda st: st[0][1]["y"] == 5     # y nunca muda
    assert iw_plan(_start(0), ["A"], _model(), goal_fn=goal, max_width=1, max_nodes=100000) is None


# --- wiring no controlador: _iw_decide usa o TypedWorldModel ---
def test_agent_iw_decide_uses_typed_model(monkeypatch):
    import numpy as np
    from agents.causal.agent import CausalObjectAgent
    from agents.causal.perception import parse, match_objects
    from agents.causal.policy import Candidate

    monkeypatch.setenv("CAUSAL_IW", "1")
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    assert a._iw_on is True
    g = np.zeros((8, 8), dtype=int); g[1, 1] = 3
    s = match_objects(None, parse(g))
    a._typed.set_rule(s.objects[0].shape_hash, _MOVE)   # regra p/ o tipo do objeto
    key = a._iw_decide(s, [Candidate(None, None, None, "ACTION1", False)])
    assert key == "ACTION1"                              # única ação, gera novidade
    # sem regras aceitas → None (fallback)
    a._typed.sources.clear(); a._typed._compiled.clear()
    assert a._iw_decide(s, [Candidate(None, None, None, "ACTION1", False)]) is None
