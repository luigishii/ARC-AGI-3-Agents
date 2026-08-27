import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.causal.agent import CausalObjectAgent


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.frames = []
    a.action_counter = 0
    a._init_causal_state()
    return a


def _frame(grid):
    return FrameData(levels_completed=0, state=GameState.NOT_FINISHED,
                     frame=[grid], available_actions=[GameAction.ACTION6.value])


def _hud_grid(tick):
    # gameplay estatico embaixo; um "contador" no topo (linha 0) que muda a cada passo
    g = np.zeros((10, 10), dtype=int)
    g[5, 5] = 3                 # objeto de gameplay ESTATICO
    g[0, tick % 10] = 4         # HUD: celula que muda de posicao a cada passo
    return g.tolist()


def test_hud_only_change_becomes_none_after_learning():
    a = _agent()
    # alimenta varias transicoes onde SO o HUD (linha 0) muda
    for t in range(8):
        a.choose_action([], _frame(_hud_grid(t)))
    # apos aprender o HUD, os efeitos observados recentes devem virar 'none'
    # (o gameplay em (5,5) nunca mudou; so a linha 0, agora mascarada)
    recent = [r["actual"] for r in a._instr.records if r["actual"] is not None]
    assert recent and recent[-1] == "none"


def test_hud_mask_is_reset_on_reset():
    a = _agent()
    for t in range(6):
        a.choose_action([], _frame(_hud_grid(t)))
    assert a._hud.total >= 5
    over = FrameData(levels_completed=0, state=GameState.GAME_OVER, frame=[_hud_grid(0)])
    a.choose_action([], over)
    assert a._hud.total == 0            # resetado
