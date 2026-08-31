"""Fix da fixacao em ACTION2: meta 'press' e tecla unica -> executa 1x e limpa,
para nao repetir a mesma acao ate GOAL_AGE_MAX. Metas 'code' (politica) persistem.
Tambem: o prompt do LLM nao pode hardcodar ACTION2 (priming que o Qwen papagaia).
"""
import numpy as np
from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent
from agents.causal import llm as llm_mod


class _Frame:
    def __init__(self, frame, avail):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = 0
        self.win_levels = 0
        self.full_reset = False
        self.available_actions = avail


def _grid():
    g = np.zeros((8, 8), dtype=int)
    g[1, 1] = 3
    return g


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_press_goal_cleared_after_one_use(monkeypatch):
    a = _agent(monkeypatch)
    a._goal = {"type": "press", "action": "ACTION2"}
    a._goal_age = 0
    a._goal_fails = 0
    action = a.choose_action([], _Frame(_grid(), [GameAction.ACTION1, GameAction.ACTION2]))
    assert action is GameAction.ACTION2      # usou a meta neste passo
    assert a._goal is None                   # mas limpou -> nao repete no proximo


def test_code_goal_persists(monkeypatch):
    a = _agent(monkeypatch)
    a._goal = {"type": "code", "source": "def decide(scene):\n    return 'ACTION2'\n"}
    a._goal_age = 0
    a._goal_fails = 0
    a.choose_action([], _Frame(_grid(), [GameAction.ACTION1, GameAction.ACTION2]))
    assert a._goal is not None                # meta 'code' e politica persistente


def test_prompt_does_not_hardcode_action2():
    # priming removido: nenhum exemplo 'press' fixa ACTION2 no instruction/fewshot
    assert '"action":"ACTION2"' not in llm_mod._INSTRUCTION
    assert "press('ACTION2')" not in llm_mod._FEWSHOT
