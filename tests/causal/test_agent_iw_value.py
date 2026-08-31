import numpy as np

from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects
from agents.causal.policy import Candidate


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def _scene():
    g = np.zeros((8, 8), dtype=int)
    g[1, 1] = 3
    return match_objects(None, parse(g))


def _cands():
    return [Candidate(None, None, None, "ACTION1", False)]


# --- _iw_decide passa value_fn (não goal_fn) e conta call+hit quando há melhoria ---
def test_iw_decide_uses_value_fn(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._typed.sources = {"shp": "def transition(obj, action, ctx): return obj"}
    a._reward_fn = lambda state: (5.0, False)
    captured = {}

    def fake_iw_plan(start, actions, model, goal_fn=None, value_fn=None,
                     max_width=2, max_nodes=1000):
        captured["goal_fn"] = goal_fn
        captured["value_fn"] = value_fn
        return "ACTION1"

    monkeypatch.setattr("agents.causal.agent.iw_plan", fake_iw_plan)
    out = a._iw_decide(_scene(), _cands())
    assert out == "ACTION1"
    assert captured["goal_fn"] is None              # não usa mais goal_fn
    assert callable(captured["value_fn"])           # passa value_fn
    assert a._iw_goal_calls == 1
    assert a._iw_goal_hits == 1


# --- sem melhoria (iw_plan devolve None): call conta, hit não ---
def test_iw_decide_value_miss(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._typed.sources = {"shp": "def transition(obj, action, ctx): return obj"}
    a._reward_fn = lambda state: (5.0, False)
    monkeypatch.setattr("agents.causal.agent.iw_plan", lambda *args, **kw: None)
    out = a._iw_decide(_scene(), _cands())
    assert out is None
    assert a._iw_goal_calls == 1
    assert a._iw_goal_hits == 0


# --- sem regras aceitas: None cedo, contadores não mexem ---
def test_iw_decide_no_rules(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._reward_fn = lambda state: (5.0, False)
    out = a._iw_decide(_scene(), _cands())
    assert out is None
    assert a._iw_goal_calls == 0
    assert a._iw_goal_hits == 0
