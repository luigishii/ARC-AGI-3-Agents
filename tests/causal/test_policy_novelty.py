from arcengine import GameAction
from agents.causal.perception import Scene
from agents.causal.causal_model import CausalModel
from agents.causal.novelty import NoveltyModel
from agents.causal.policy import Policy, Candidate


def _c(key, has_object=False):
    return Candidate(GameAction.ACTION6, 5, 5, key, has_object)


def test_untried_key_scores_three_parity_with_v3():
    p = Policy(seed=0)
    model = CausalModel()
    nov = NoveltyModel()
    # chave inédita: y=1.0, ctrl=1.0 → 3.0 (igual ao termo v3 de ação nova)
    assert p.score(_c("K"), model, set(), 0.0, novelty=nov) == 3.0


def test_controllability_gate_prefers_reproducible():
    p = Policy(seed=0)
    nov = NoveltyModel()
    # chave A: efeito único reprodutível (conf 1.0); chave B: dois efeitos (conf 0.5)
    hi = CausalModel()
    hi.rules["A"] = {"moved:x": 4}
    lo = CausalModel()
    lo.rules["B"] = {"moved:x": 1, "recolored:y": 1}
    # yields otimistas iguais (1.0) → só o ctrl (conf) difere
    sa = p.score(_c("A"), hi, set(), 0.0, novelty=nov)
    sb = p.score(_c("B"), lo, set(), 0.0, novelty=nov)
    assert sa > sb


def test_none_key_penalty_applied():
    p = Policy(seed=0)
    nov = NoveltyModel()
    model = CausalModel()
    model.rules["K"] = {"none:None": 5}       # sempre none, conf 1.0
    # y otimista 1.0, ctrl=conf=1.0 → +3; −2 do none → 1.0. (Com uso real o yield
    # despenca por revisita e o score fica negativo; aqui validamos o termo −2.)
    assert p.score(_c("K"), model, set(), 0.0, novelty=nov) == 1.0


def test_novelty_none_reproduces_v3():
    p = Policy(seed=0)
    model = CausalModel()
    # sem novelty: chave inédita → +3 (caminho v3)
    assert p.score(_c("K"), model, set(), 0.0) == 3.0


def test_decide_accepts_novelty_kwarg():
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    nov = NoveltyModel()
    scene = Scene(objects=[], grid=None)
    cand = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0, novelty=nov)
    assert cand is not None
