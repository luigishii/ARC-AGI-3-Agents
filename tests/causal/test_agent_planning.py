from arcengine import GameAction, GameState
from agents.causal.agent import CausalObjectAgent
from agents.causal.planning import TransitionModel


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


def _agent(monkeypatch, plan_env=None):
    if plan_env is None:
        monkeypatch.delenv("CAUSAL_PLAN", raising=False)
    else:
        monkeypatch.setenv("CAUSAL_PLAN", plan_env)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._cleanup = False
    a._init_causal_state()
    return a


def test_agent_has_transition_model(monkeypatch):
    a = _agent(monkeypatch)
    assert isinstance(a._tmodel, TransitionModel)
    assert a._plan_on is True


def test_agent_learns_transitions(monkeypatch):
    a = _agent(monkeypatch)
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))       # fecha loop → observa 1 transição
    assert len(a._tmodel.trans) >= 1


def test_plan_disabled_reproduces_v7(monkeypatch):
    a = _agent(monkeypatch, plan_env="0")
    assert a._plan_on is False
    act = a.choose_action([], _Frame(_grid(3)))   # não estoura; retorna ação
    assert act is not None
