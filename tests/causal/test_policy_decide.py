# tests/causal/test_policy_decide.py
import numpy as np
from arcengine import GameAction

from agents.causal.perception import parse, match_objects
from agents.causal.causal_model import CausalModel, Effect
from agents.causal.policy import Policy


def _scene():
    g = np.zeros((6, 6), dtype=int)
    g[1, 1] = 3
    return match_objects(None, parse(g.tolist()))


def test_prefers_untried_over_known_none():
    m = CausalModel()
    m._bump("ACTION1", Effect("none", None))          # conhecido, inútil
    p = Policy(seed=1, epsilon=0.0)
    cand = p.decide(_scene(), m, [GameAction.ACTION1, GameAction.ACTION2], set(), budget_frac=1.0)
    assert cand.action is GameAction.ACTION2          # inédito ganha


def test_prefers_progress_action():
    m = CausalModel()
    m._bump("ACTION1", Effect("structural", ("disappeared",)), level_up=True)  # progresso!
    m._bump("ACTION2", Effect("moved", (0, 1)))
    p = Policy(seed=1, epsilon=0.0)
    cand = p.decide(_scene(), m, [GameAction.ACTION1, GameAction.ACTION2], set(), budget_frac=0.2)
    assert cand.action is GameAction.ACTION1


def test_epsilon_zero_is_deterministic():
    m = CausalModel()
    p = Policy(seed=1, epsilon=0.0)
    args = (_scene(), m, [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3], set(), 1.0)
    assert p.decide(*args).action is p.decide(*args).action


def test_decide_accepts_int_ids_for_available_actions():
    # available_actions costuma chegar como list[int] (ids) vindo do harness;
    # decide()/candidates() devem normalizar via GameAction.from_id.
    m = CausalModel()
    p = Policy(seed=1, epsilon=0.0)
    cand = p.decide(_scene(), m, [1, 6], set(), budget_frac=1.0)
    assert cand is not None
    assert cand.action in (GameAction.ACTION1, GameAction.ACTION6)
    assert isinstance(cand.action, GameAction)
