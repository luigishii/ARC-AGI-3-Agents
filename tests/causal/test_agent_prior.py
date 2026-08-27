from arcengine import GameAction, GameState
from agents.causal.agent import CausalObjectAgent
from agents.causal.transfer import TransferPrior, shared_prior, reset_shared_prior


class _Frame:
    def __init__(self, frame, state=GameState.NOT_FINISHED, levels=0):
        self.frame = frame
        self.state = state
        self.levels_completed = levels
        self.available_actions = [GameAction.ACTION1]
        self.full_reset = False


def _grid(v):
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = v
    return [g]


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_agent_uses_shared_prior_instance():
    reset_shared_prior()
    a = _agent()
    assert a._prior is shared_prior()
    assert isinstance(a._prior, TransferPrior)


def test_agent_feeds_prior_over_steps():
    reset_shared_prior()
    a = _agent()
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))     # fecha o loop da 1ª ação → observe
    total = sum(v[1] for v in a._prior._counts.values())
    assert total >= 1


def test_two_agents_share_same_prior():
    reset_shared_prior()
    a = _agent()
    b = _agent()
    assert a._prior is b._prior


def test_reset_does_not_wipe_prior():
    reset_shared_prior()
    a = _agent()
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))
    a._prior.observe("simple", "moved")
    before = sum(v[1] for v in a._prior._counts.values())
    a.choose_action([], _Frame(_grid(3), state=GameState.GAME_OVER))  # RESET
    after = sum(v[1] for v in a._prior._counts.values())
    assert after >= before
