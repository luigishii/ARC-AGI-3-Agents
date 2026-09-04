"""Item 1: cap GLOBAL de chamadas LLM por processo (todas as threads do Swarm)."""
from arcengine import GameAction, GameState

import agents.causal.agent as agent_mod
from agents.causal.agent import CausalObjectAgent


class _Fake:
    def __init__(self, canned):
        self.canned, self.calls = canned, 0

    def complete(self, prompt):
        self.calls += 1
        return self.canned


class _Frame:
    def __init__(self, frame, available=None):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = 0
        self.available_actions = available or [GameAction.ACTION1]
        self.full_reset = False


def _grid():
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = 3
    return [g]


def _agent(monkeypatch, game_id="test", **env):
    env.setdefault("CAUSAL_LLM", "1")
    env.setdefault("CAUSAL_LLM_DEFER", "0")
    env.setdefault("CAUSAL_MAX_ACTIONS", "10000")
    env.setdefault("CAUSAL_DIRECT", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 10
    a.game_id = game_id
    a.MAX_ACTIONS = 10000
    a._init_causal_state()
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    return a


def test_global_cap_shared_across_agents(monkeypatch):
    monkeypatch.setitem(agent_mod._LLM_TOTAL, "calls", 0)
    monkeypatch.setenv("CAUSAL_LLM_TOTAL_CALLS", "1")
    a, b = _agent(monkeypatch, "g1"), _agent(monkeypatch, "g2")
    a.choose_action([], _Frame(_grid()))
    b.choose_action([], _Frame(_grid()))
    assert a._llm.calls == 1
    assert b._llm.calls == 0                       # cap global atingido
    assert agent_mod._LLM_TOTAL["calls"] == 1
    assert b.phase2_stats()["llm_total_calls"] == 1


def test_global_cap_unlimited_by_default(monkeypatch):
    monkeypatch.setitem(agent_mod._LLM_TOTAL, "calls", 0)
    monkeypatch.delenv("CAUSAL_LLM_TOTAL_CALLS", raising=False)
    a, b = _agent(monkeypatch, "g1"), _agent(monkeypatch, "g2")
    a.choose_action([], _Frame(_grid()))
    b.choose_action([], _Frame(_grid()))
    assert a._llm.calls == 1 and b._llm.calls == 1


def test_global_cap_also_gates_class_inference(monkeypatch):
    monkeypatch.setitem(agent_mod._LLM_TOTAL, "calls", 5)
    monkeypatch.setenv("CAUSAL_LLM_TOTAL_CALLS", "5")
    a = _agent(monkeypatch, "zzzz", CAUSAL_CLASS="1", CAUSAL_DIRECT="0")
    a._llm = _Fake('{"cls":"A","avatar":9,"target":5,"click":[9],"hud_rows":[],"hud_cols":[]}')
    a.choose_action([], _Frame(_grid()))
    assert a._llm.calls == 0 and a._gk == {}
