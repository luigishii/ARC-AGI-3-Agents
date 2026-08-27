import numpy as np
from arcengine import GameAction, GameState

from agents.causal.causal_model import CausalModel, Effect
from agents.causal.agent import CausalObjectAgent


# --- Gap 3: grading-set exclui no-op↔no-op ---
def test_noop_pair_excluded_from_grading():
    m = CausalModel()
    for _ in range(100):
        m.record_prediction(Effect("none", None), Effect("none", None))  # não pontua
    assert m.stats()["prediction_accuracy"] == 0.0     # 0/0 → 0, não inflado


def test_false_movement_on_real_noop_is_penalized():
    m = CausalModel()
    m.record_prediction(Effect("moved", (0, 1)), Effect("none", None))   # prevê falso movimento
    assert m.stats()["prediction_accuracy"] == 0.0     # 0 acertos, 1 graded


def test_real_change_still_graded():
    m = CausalModel()
    m.record_prediction(Effect("moved", (0, 1)), Effect("moved", (0, 1)))  # acerto
    m.record_prediction(Effect("none", None), Effect("moved", (0, 1)))     # erro (graded)
    assert abs(m.stats()["prediction_accuracy"] - 0.5) < 1e-9


# --- Gap 2: transição de level-up não polui os aprendizes de dinâmica ---
class _Frame:
    def __init__(self, frame, levels=0):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = levels
        self.available_actions = [GameAction.ACTION1]
        self.full_reset = False


def _grid(col):
    g = np.zeros((8, 8), dtype=int)
    g[1, col] = 3
    return [g.tolist()]


def _agent(monkeypatch):
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_levelup_transition_not_fed_to_dynamics(monkeypatch):
    a = _agent(monkeypatch)
    a.choose_action([], _Frame(_grid(1), levels=0))    # passo 1: sem close-loop
    n_buf0, n_types0 = len(a._buffer), len(a._type_buffer)
    a.choose_action([], _Frame(_grid(2), levels=1))    # passo 2: LEVEL-UP no close-loop
    # o sucessor é init-de-próximo-nível → NÃO entra nos aprendizes de dinâmica
    assert len(a._buffer) == n_buf0
    assert len(a._type_buffer) == n_types0
    assert a._tmodel.trans == {}
    assert a._novelty.goal_anchors            # mas o desfecho vira âncora de meta


def test_normal_transition_is_fed_to_dynamics(monkeypatch):
    a = _agent(monkeypatch)
    a.choose_action([], _Frame(_grid(1), levels=0))
    a.choose_action([], _Frame(_grid(2), levels=0))    # sem level-up → decisão→decisão
    assert len(a._buffer) == 1                          # transição mecânica registrada
