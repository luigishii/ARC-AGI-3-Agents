"""Telemetria: qual camada da pilha decidiu cada acao (reasoning['layer'] + stats)."""
from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent


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


def _agent(monkeypatch, **env):
    env.setdefault("CAUSAL_MAX_ACTIONS", "10000")
    env.setdefault("CAUSAL_LLM", "0")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 20          # depois do sweep
    a.game_id = "test"
    a.MAX_ACTIONS = 10000
    a._init_causal_state()
    return a


def test_cover_layer_labelled_and_counted(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_COVER="1", CAUSAL_FIX="0")
    act = a.choose_action([], _Frame(_grid(), [GameAction.ACTION1, GameAction.ACTION2]))
    assert act.reasoning["layer"] == "cover"
    assert a.phase2_stats()["layers"] == {"cover": 1}


def test_greedy_when_all_keys_tried(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_COVER="1", CAUSAL_FIX="0")
    a._cover = {"ACTION1": 1, "ACTION2": 1}
    act = a.choose_action([], _Frame(_grid(), [GameAction.ACTION1, GameAction.ACTION2]))
    assert act.reasoning["layer"] == "greedy"   # cobertura cedeu (tudo ja tentado)


def test_effective_keys_layer_labelled(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_COVER="1", CAUSAL_FIX="0")
    a._effective_keys.append("ACTION2")
    act = a.choose_action([], _Frame(_grid(), [GameAction.ACTION1, GameAction.ACTION2]))
    assert act.name == "ACTION2"
    assert act.reasoning["layer"] == "effkeys"
