from dataclasses import replace
from agents.causal.perception import Scene, Object
from agents.causal.causal_model import CausalModel
from agents.causal.planning import TransitionModel
from agents.causal.novelty import NoveltyModel, state_signature
from agents.causal.ranker import rank_candidates


def _obj(cells, color=3, oid=0):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return replace(Object(color, cset, (0, 0, 0, 0), centroid, 1, "h"), id=oid)


def _scene():
    return Scene(objects=[_obj([(0, 0)], oid=1)], grid=None)


def test_rank_picks_higher_novelty_action():
    scene = _scene()
    sig = state_signature(scene)
    tmodel = TransitionModel()
    tmodel.observe(sig, "ACTION1", "A1")     # inédito → alta novidade
    tmodel.observe(sig, "ACTION2", "A2")
    nov = NoveltyModel()
    for _ in range(50):
        nov.visit("A2")                       # A2 saturado → baixa novidade
    src_a = "def decide(scene):\n    return 'ACTION1'\n"
    src_b = "def decide(scene):\n    return 'ACTION2'\n"
    best = rank_candidates([src_b, src_a], scene, CausalModel(), tmodel, nov, {})
    assert best == src_a


def test_rank_discards_invalid_and_returns_none():
    scene = _scene()
    bad = "def decide(scene):\n    return 1/0\n"          # erro → descartado
    worse = "def decide(scene) syntax"                    # não compila
    assert rank_candidates([bad, worse], scene, CausalModel(),
                           TransitionModel(), NoveltyModel(), {}) is None


def test_rank_uses_dsl_in_candidates():
    scene = _scene()
    src = "def decide(scene):\n    return click(2, 3)\n"  # usa a DSL injetada
    best = rank_candidates([src], scene, CausalModel(),
                           TransitionModel(), NoveltyModel(), {})
    assert best == src
