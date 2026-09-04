"""plan(): na FRONTEIRA (transicao desconhecida) desempata pela produtividade aprendida
da chave (CausalModel/TransferPrior) em vez da ordem da lista (raster)."""
from agents.causal.planning import TransitionModel, plan
from agents.causal.novelty import NoveltyModel
from agents.causal.agent import CausalObjectAgent
from agents.causal.policy import Candidate
from agents.causal.perception import Scene
from types import SimpleNamespace


def _setup():
    tm, nv = TransitionModel(), NoveltyModel()
    s0, s1 = (("a", 0, 0),), (("a", 1, 0),)
    tm.observe(s0, "A", s1)          # A conhecida; B e C = fronteira
    nv.visit(s1); nv.visit(s1)       # s1 ja visitado -> A menos nova que a fronteira
    return tm, nv, s0


def test_plan_without_prior_keeps_list_order():
    tm, nv, s0 = _setup()
    assert plan(s0, ["A", "B", "C"], tm, nv, []) == "B"


def test_plan_frontier_prefers_productive_key():
    tm, nv, s0 = _setup()
    prior = {"B": 0.0, "C": 1.0}.get
    assert plan(s0, ["A", "B", "C"], tm, nv, [], key_prior=prior) == "C"


def test_plan_known_inert_key_loses_to_unknown():
    tm, nv, s0 = _setup()
    prior = {"B": 0.0, "C": 0.5}.get      # B ja vista como inerte; C nunca vista (neutra)
    assert plan(s0, ["A", "B", "C"], tm, nv, [], key_prior=prior) == "C"


def _agent(monkeypatch):
    monkeypatch.setenv("CAUSAL_MAX_ACTIONS", "1000")
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.game_id = "t"
    a.MAX_ACTIONS = 1000
    a._init_causal_state()
    return a


def _obj(color=9):
    return SimpleNamespace(id=0, color=color, centroid=(1, 1), size=4, bbox=(1, 1, 2, 2),
                           cells={(1, 1)}, shape_hash="h")


def test_key_productivity_from_model_and_prior(monkeypatch):
    a = _agent(monkeypatch)
    from agents.causal.causal_model import Effect
    a._model._bump("inert", Effect("none", None))
    a._model._bump("inert", Effect("none", None))
    a._model._bump("live", Effect("structural", ("x",)))
    a._model._bump("live", Effect("none", None))
    kb = Candidate(None, None, None, "kb", False)
    assert a._key_productivity("inert", kb) == 0.0
    assert a._key_productivity("live", kb) == 0.5
    # nunca vista -> produtividade do prior pela feature abstrata (0..1)
    p = a._key_productivity("never", kb)
    assert 0.0 <= p <= 1.0


def test_two_phase_click_click_skips_known_inert(monkeypatch):
    """two-phase clique->clique: entre os alternativos, evita chave ja vista como inerte."""
    from agents.causal.causal_model import Effect
    a = _agent(monkeypatch)
    a._model._bump("ACTION6@c9s2@0,0", Effect("none", None))
    a._model._bump("ACTION6@c9s2@0,0", Effect("none", None))
    a._cover = {"ACTION6@c9s2@0,0": 1, "ACTION6@c9s2@3,3": 2}   # o inerte e o MENOS visitado
    inert = Candidate(None, 5, 5, "ACTION6@c9s2@0,0", True, 36)
    live = Candidate(None, 40, 40, "ACTION6@c9s2@3,3", True, 36)
    assert a._two_phase_click_pick([inert, live]) == "ACTION6@c9s2@3,3"
