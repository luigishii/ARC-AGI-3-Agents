import numpy as np
from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent
from agents.causal.novelty import state_signature
from agents.causal.perception import parse, to_grid, win_grid


def test_win_grid_returns_win_layer_not_next_level():
    # No passo que completa um nivel, o frame empilha DUAS camadas:
    # [tabuleiro de VITORIA do nivel vencido, init do PROXIMO nivel].
    win_board = np.zeros((8, 8), dtype=int)
    win_board[1, 5] = 3
    next_init = np.zeros((8, 8), dtype=int)
    next_init[6, 2] = 7
    frame = [win_board.tolist(), next_init.tolist()]
    # to_grid (percepcao normal) pega a ULTIMA camada = init do proximo nivel
    assert np.array_equal(to_grid(frame), next_init)
    # win_grid recupera a camada de VITORIA (a que to_grid descartava)
    assert np.array_equal(win_grid(frame), win_board)


def test_win_grid_single_layer_falls_back_to_board():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = 2
    assert np.array_equal(win_grid([board.tolist()]), board)  # 1 camada empilhada
    assert np.array_equal(win_grid(board.tolist()), board)    # grid cru (sem pilha)


# --- integracao: no level-up a ancora de meta = tabuleiro RESOLVIDO (camada de vitoria) ---
class _Frame:
    def __init__(self, frame, levels=0):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = levels
        self.available_actions = [GameAction.ACTION1]
        self.full_reset = False


def _board(col):
    # grade real 64x64: `cell_of` bina por gx=col*6//64, entao colunas separadas
    # caem em celulas de grade distintas (state_signature vira distinguivel).
    g = np.zeros((64, 64), dtype=int)
    g[1, col] = 3
    return g.tolist()


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_levelup_anchor_is_win_board_not_next_level_init():
    a = _agent()
    a.choose_action([], _Frame([_board(55)], levels=0))  # decisao pre-vitoria (gx=5)
    # LEVEL-UP: frame empilha [tabuleiro de VITORIA (col40=gx3), init proximo (col8=gx0)]
    win, nxt = _board(40), _board(8)
    a.choose_action([], _Frame([win, nxt], levels=1))
    sig_win = state_signature(parse(win_grid([win, nxt]), hud_mask=a._hud.mask()))
    sig_next = state_signature(parse(nxt, hud_mask=a._hud.mask()))
    # ancorou o estado RESOLVIDO (camada de vitoria), nao o init do proximo nivel
    assert a._novelty.goal_anchors == [sig_win]
    assert sig_next not in a._novelty.goal_anchors
