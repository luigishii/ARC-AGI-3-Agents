import json

import numpy as np

from agents.causal.goals import (
    compile_reward, static_reward_check, goal_fn_from_reward,
)
from agents.causal.llm import build_prompt
from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects
from agents.causal.policy import Candidate


_REWARD_OK = (
    "def reward_function(state):\n"
    "    goal = any(o[1].get('x', 0) >= 5 for o in state)\n"
    "    return (1.0 if goal else 0.0, goal)\n"
)
_REWARD_IGNORES_STATE = (
    "def reward_function(state):\n"
    "    return (1.0, True)\n"           # não olha o state → trapaça
)
_REWARD_GLOBAL = (
    "def reward_function(state):\n"
    "    global CACHE\n"
    "    return (1.0, CACHE)\n"
)


# --- A: static_reward_check (anti-trapaça estilo OPINE) ---
def test_static_check_accepts_state_predicate():
    assert static_reward_check(_REWARD_OK) is True


def test_static_check_rejects_state_ignoring():
    assert static_reward_check(_REWARD_IGNORES_STATE) is False


def test_static_check_rejects_global_cheat():
    assert static_reward_check(_REWARD_GLOBAL) is False


def test_static_check_rejects_non_compiling():
    assert static_reward_check("def reward_function(state) return") is False


# --- A: goal_fn_from_reward ---
def test_goal_fn_reads_goal_flag():
    fn = compile_reward(_REWARD_OK)
    goal = goal_fn_from_reward(fn)
    assert goal([("t", {"x": 5})]) is True
    assert goal([("t", {"x": 0})]) is False


# --- C: build_prompt traz few-shot + available_actions ---
def test_build_prompt_has_fewshot_and_actions():
    scene = match_objects(None, parse(np.zeros((8, 8), dtype=int)))
    p = build_prompt(scene, {"available": ["ACTION1", "ACTION2"], "moves": {}, "notes": ""})
    assert "def decide(scene)" in p          # exemplo few-shot de código
    assert "ACTION1" in p and "ACTION2" in p  # ações disponíveis no prompt


# --- harness p/ o agente ---
class _Fake:
    def __init__(self, canned):
        self.canned = canned

    def complete(self, prompt):
        return self.canned


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


# --- A: agente aprende reward_function e IW vira goal-directed ---
def test_agent_learns_reward(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_TYPED="1")
    a._llm = _Fake(json.dumps({"type": "code", "source": _REWARD_OK}))
    s = match_objects(None, parse(np.zeros((8, 8), dtype=int)))
    assert a._try_learn_reward(s) is True
    assert a._reward_fn is not None


def test_iw_goal_directed_when_reward_present(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    g = np.zeros((8, 8), dtype=int); g[1, 1] = 3
    s = match_objects(None, parse(g))
    tau = s.objects[0].shape_hash
    a._typed.set_rule(tau, "def transition(obj, action, ctx):\n    o=dict(obj)\n    o['x']=o['x']+1\n    return o\n")
    a._reward_fn = compile_reward(
        "def reward_function(state):\n"
        "    goal = any(o[1].get('x', 0) >= 3 for o in state)\n"
        "    return (1.0 if goal else 0.0, goal)\n"
    )
    key = a._iw_decide(s, [Candidate(None, None, None, "ACTION1", False)])
    assert key == "ACTION1"       # planeja rumo ao goal x>=3


# --- B: telemetria da Fase-2 ---
def test_phase2_stats_keys(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    st = a.phase2_stats()
    assert st["llm_kind"] == "null"
    assert st["n_rules"] == 0
    assert st["reward_learned"] is False
    assert "llm_calls" in st and "n_types" in st
