from arcengine import GameAction, GameState
from agents.causal.agent import CausalObjectAgent
from agents.causal.novelty import NoveltyModel


class _Frame:
    def __init__(self, frame, state=GameState.NOT_FINISHED, levels=0):
        self.frame = frame
        self.state = state
        self.levels_completed = levels
        self.available_actions = [GameAction.ACTION1]
        self.full_reset = False


def _grid(v):
    # pilha 1×8×8; célula (1,1) = v, resto 0 (fundo)
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = v
    return [g]


def test_agent_accumulates_novelty_over_steps():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    assert isinstance(a._novelty, NoveltyModel)
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))   # muda estado
    # após fechar o loop da 1ª ação, houve ao menos uma transição observada
    assert sum(v[1] for v in a._novelty._yield.values()) >= 1


def test_reset_does_not_wipe_novelty():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))
    a._novelty.counts["x"] = 7
    a.choose_action([], _Frame(_grid(3), state=GameState.GAME_OVER))  # RESET
    assert a._novelty.counts.get("x") == 7


def test_level_up_records_goal_anchor():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    a.choose_action([], _Frame(_grid(3), levels=0))
    a.choose_action([], _Frame(_grid(4), levels=1))   # level up no passo seguinte
    assert len(a._novelty.goal_anchors) >= 1
