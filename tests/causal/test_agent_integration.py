import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.causal.agent import CausalObjectAgent


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.frames = []
    a.action_counter = 0
    a._init_causal_state()
    return a


def _frame(grid, level=0, actions=(GameAction.ACTION1, GameAction.ACTION2)):
    return FrameData(
        levels_completed=level,
        state=GameState.NOT_FINISHED,
        frame=[grid],
        available_actions=list(actions),
    )


def test_closes_causal_loop_across_two_steps():
    a = _agent()
    g0 = np.zeros((6, 6), dtype=int); g0[1, 1] = 3
    f0 = _frame(g0.tolist())
    act0 = a.choose_action([f0], f0)          # 1o passo: sem transicao ainda
    assert act0.value in f0.available_actions   # available_actions sao ints (ids)
    g1 = np.zeros((6, 6), dtype=int); g1[1, 2] = 3   # objeto moveu
    f1 = _frame(g1.tolist())
    a.choose_action([f0, f1], f1)             # observa a transicao de act0
    assert a._model.stats()["coverage_keys"] >= 1


def test_deferred_log_records_observed_effect():
    a = _agent()
    g0 = np.zeros((6, 6), dtype=int); g0[1, 1] = 3
    f0 = _frame(g0.tolist())
    a.choose_action([f0], f0)                  # 1o passo: log fica pendente, nada gravado
    assert a._instr.records == []             # nada logado antes de observar efeito
    g1 = np.zeros((6, 6), dtype=int); g1[1, 2] = 3   # objeto moveu -> efeito "moved"
    f1 = _frame(g1.tolist())
    a.choose_action([f0, f1], f1)             # fecha loop -> loga a acao do 1o passo com actual real
    filled = [r for r in a._instr.records if r["actual"] is not None]
    assert len(filled) >= 1
    assert filled[0]["actual"] == "moved"


def test_detects_level_up_as_progress():
    a = _agent()
    g0 = np.zeros((6, 6), dtype=int); g0[1, 1] = 3
    f0 = _frame(g0.tolist(), level=0)
    a.choose_action([f0], f0)
    g1 = np.zeros((6, 6), dtype=int); g1[5, 5] = 4
    f1 = _frame(g1.tolist(), level=1)         # levels_completed subiu
    a.choose_action([f0, f1], f1)
    assert len(a._model.progress_keys) >= 1
