"""Regressao: full_reset=True NAO deve re-enviar RESET (offline loopava infinito).

Offline, o jogo local devolve full_reset=True na resposta de TODO RESET. O agente
usava full_reset tanto p/ limpar memoria QUANTO p/ decidir RESET -> RESET gerava
full_reset -> RESET ... loop infinito (200 acoes, 0 chamadas de LLM). O fix: full_reset
so limpa as caches; RESET de verdade so quando state in (NOT_PLAYED, GAME_OVER).
"""
import numpy as np
from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent


class _Frame:
    def __init__(self, frame, state=GameState.NOT_FINISHED, full_reset=False):
        self.frame = frame
        self.state = state
        self.levels_completed = 0
        self.win_levels = 0
        self.full_reset = full_reset
        self.available_actions = [GameAction.ACTION1]


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


def test_full_reset_with_playable_state_does_not_reset(monkeypatch):
    # full_reset=True mas o jogo esta jogavel (NOT_FINISHED) -> deve JOGAR, nao RESET.
    a = _agent(monkeypatch)
    action = a.choose_action([], _Frame(_grid(), full_reset=True))
    assert action is not GameAction.RESET


def test_not_played_still_resets(monkeypatch):
    a = _agent(monkeypatch)
    action = a.choose_action([], _Frame(_grid(), state=GameState.NOT_PLAYED))
    assert action is GameAction.RESET


def test_game_over_still_resets(monkeypatch):
    a = _agent(monkeypatch)
    action = a.choose_action([], _Frame(_grid(), state=GameState.GAME_OVER))
    assert action is GameAction.RESET
