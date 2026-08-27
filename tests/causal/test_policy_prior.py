from arcengine import GameAction
from agents.causal.perception import Scene
from agents.causal.causal_model import CausalModel
from agents.causal.transfer import TransferPrior
from agents.causal.policy import Policy, Candidate


def _click(has_object, key="ACTION6@cell=0,0"):
    return Candidate(GameAction.ACTION6, 5, 5, key, has_object)


def test_prior_boosts_productive_feature():
    p = Policy(seed=0)
    model = CausalModel()
    prior = TransferPrior()
    # torna "click_on_object" muito produtivo e "click_empty" improdutivo
    for _ in range(9):
        prior.observe("click_on_object", "moved")
    prior.observe("click_on_object", "none")
    for _ in range(9):
        prior.observe("click_empty", "none")
    prior.observe("click_empty", "moved")
    s_obj = p.score(_click(True), model, set(), 0.0, prior=prior)
    s_empty = p.score(_click(False), model, set(), 0.0, prior=prior)
    assert s_obj > s_empty


def test_prior_none_reproduces_v4():
    p = Policy(seed=0)
    model = CausalModel()
    # sem prior nem novelty: chave inédita → +3 (caminho v4/v3), has_object=False
    assert p.score(_click(False), model, set(), 0.0) == 3.0


def test_prior_term_magnitude():
    p = Policy(seed=0)
    model = CausalModel()
    prior = TransferPrior()          # neutro 0.5 → termo +0.5
    # chave inédita: base 3.0 (eff None → +3) + prior 0.5 = 3.5 (has_object False)
    assert p.score(_click(False), model, set(), 0.0, prior=prior) == 3.5


def test_decide_accepts_prior_kwarg():
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    prior = TransferPrior()
    scene = Scene(objects=[], grid=None)
    cand = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0, prior=prior)
    assert cand is not None
