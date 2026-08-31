import json

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


def _grid_at(col):
    g = np.zeros((8, 8), dtype=int)
    g[1, col] = 3
    return g


def _scene():
    return match_objects(None, parse(_grid_at(1)))


def _cands():
    return [Candidate(None, None, None, "ACTION1", False)]


# --- _iw_decide conta call+hit quando o IW acha caminho até a meta ---
def test_iw_decide_counts_goal_hit(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._typed.sources = {"shp": "def transition(obj, action, ctx): return obj"}
    a._reward_fn = lambda state: (0.0, True)
    monkeypatch.setattr("agents.causal.agent.iw_plan",
                        lambda *args, **kw: "ACTION1")
    out = a._iw_decide(_scene(), _cands())
    assert out == "ACTION1"
    assert a._iw_goal_calls == 1
    assert a._iw_goal_hits == 1


# --- _iw_decide conta call mas NÃO hit quando o IW não acha caminho (None) ---
def test_iw_decide_counts_goal_miss(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._typed.sources = {"shp": "def transition(obj, action, ctx): return obj"}
    a._reward_fn = lambda state: (0.0, False)
    monkeypatch.setattr("agents.causal.agent.iw_plan",
                        lambda *args, **kw: None)
    out = a._iw_decide(_scene(), _cands())
    assert out is None
    assert a._iw_goal_calls == 1
    assert a._iw_goal_hits == 0


# --- sem regras aceitas: retorna None cedo, gf nunca montado, contadores não mexem ---
def test_iw_decide_no_rules_no_count(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._reward_fn = lambda state: (0.0, True)   # há reward, mas não há f_τ
    out = a._iw_decide(_scene(), _cands())
    assert out is None
    assert a._iw_goal_calls == 0
    assert a._iw_goal_hits == 0


# --- phase2_stats expõe as 5 chaves novas + reward_src ---
def test_phase2_stats_has_diag_keys(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_src = "def reward_function(state): return (0, False)"
    a._iw_goal_calls = 4
    a._iw_goal_hits = 1
    a._reward_real_true = 2
    a._reward_real_evals = 9
    s = a.phase2_stats()
    assert s["reward_src"] == "def reward_function(state): return (0, False)"
    assert s["iw_goal_calls"] == 4
    assert s["iw_goal_hits"] == 1
    assert s["reward_real_true"] == 2
    assert s["reward_real_evals"] == 9
    # continua serializável em JSON (grava em causal_phase2.json)
    json.dumps(s)


# --- _eval_reward_real conta eval+true quando o predicado dá goal_flag=True ---
def test_eval_reward_real_true(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_fn = lambda state: (1.0, True)
    a._eval_reward_real(_scene())
    assert a._reward_real_evals == 1
    assert a._reward_real_true == 1


# --- goal_flag=False: conta eval mas não true ---
def test_eval_reward_real_false(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_fn = lambda state: (0.0, False)
    a._eval_reward_real(_scene())
    assert a._reward_real_evals == 1
    assert a._reward_real_true == 0


# --- sem reward_fn: não avalia nada ---
def test_eval_reward_real_none(monkeypatch):
    a = _agent(monkeypatch)
    assert a._reward_fn is None
    a._eval_reward_real(_scene())
    assert a._reward_real_evals == 0
    assert a._reward_real_true == 0


# --- predicado que quebra não derruba nem conta true (exception-safe) ---
def test_eval_reward_real_exception_safe(monkeypatch):
    a = _agent(monkeypatch)
    def _boom(state):
        raise ValueError("boom")
    a._reward_fn = _boom
    a._eval_reward_real(_scene())        # não levanta
    assert a._reward_real_evals == 1
    assert a._reward_real_true == 0
