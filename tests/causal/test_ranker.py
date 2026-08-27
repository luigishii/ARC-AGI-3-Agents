from dataclasses import replace
from agents.causal.perception import Scene, Object
from agents.causal.causal_model import CausalModel
from agents.causal.planning import TransitionModel
from agents.causal.novelty import NoveltyModel, state_signature
from agents.causal.ranker import rank_candidates, retrodiction_score
from agents.causal.dsl import DSL


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


# --- §1: retrodiction_score ---
def test_retrodiction_empty_buffer_is_neutral():
    scene = _scene()
    src = "def decide(scene):\n    return 'ACTION1'\n"
    assert retrodiction_score(src, [], {**DSL, "MOVES": {}}) == 0.0


def test_retrodiction_positive_for_reproducing_productive_action():
    scene = _scene()
    src = "def decide(scene):\n    return 'ACTION1'\n"
    buf = [(scene, "ACTION1", "moved")]           # ação tomada foi produtiva
    assert retrodiction_score(src, buf, {**DSL, "MOVES": {}}) == 1.0


def test_retrodiction_negative_for_reproducing_dead_action():
    scene = _scene()
    src = "def decide(scene):\n    return 'ACTION1'\n"
    buf = [(scene, "ACTION1", "none")]            # ação tomada foi morta
    assert retrodiction_score(src, buf, {**DSL, "MOVES": {}}) == -1.0


def test_retrodiction_neutral_when_no_overlap_with_history():
    scene = _scene()
    src = "def decide(scene):\n    return 'ACTION1'\n"
    buf = [(scene, "ACTION2", "moved")]           # candidato nunca casa
    assert retrodiction_score(src, buf, {**DSL, "MOVES": {}}) == 0.0


# --- §1: integração no ranker ---
def test_rank_prefers_retrodiction_positive_over_equal_novelty():
    scene = _scene()
    a = "def decide(scene):\n    return 'ACTION1'\n"
    b = "def decide(scene):\n    return 'ACTION2'\n"
    # novidade idêntica (tmodel/novelty vazios → nov=1 p/ ambos), mas o buffer
    # mostra ACTION1 produtiva e ACTION2 morta
    buf = [(scene, "ACTION1", "moved"), (scene, "ACTION2", "none")]
    best = rank_candidates([b, a], scene, CausalModel(),
                           TransitionModel(), NoveltyModel(), {}, buffer=buf)
    assert best == a


# --- §1: trava de available_actions ---
def test_rank_discards_unavailable_action():
    scene = _scene()
    a = "def decide(scene):\n    return 'ACTION1'\n"   # indisponível
    b = "def decide(scene):\n    return 'ACTION2'\n"   # disponível
    best = rank_candidates([a, b], scene, CausalModel(),
                           TransitionModel(), NoveltyModel(), {},
                           available=["ACTION2"])
    assert best == b


# --- §1: deduplicação semântica ---
def test_rank_dedups_candidates_producing_same_action():
    scene = _scene()
    a = "def decide(scene):\n    return 'ACTION1'\n"
    b = "def decide(scene):\n    return 'ACTIO' + 'N1'\n"   # mesma ação resultante
    # ambos produzem ACTION1 → só o 1º é considerado
    best = rank_candidates([a, b], scene, CausalModel(),
                           TransitionModel(), NoveltyModel(), {})
    assert best == a


# --- §3: penalização por no-op ---
def test_rank_penalizes_known_noop_action():
    scene = _scene()
    sig = state_signature(scene)
    model = CausalModel()
    # ACTION1 sabidamente morta (efeito 'none'); ACTION2 produtiva
    from agents.causal.causal_model import Effect
    for _ in range(5):
        model._bump("ACTION1", Effect("none", None))
        model._bump("ACTION2", Effect("moved", (1, 0)))
    a = "def decide(scene):\n    return 'ACTION1'\n"
    b = "def decide(scene):\n    return 'ACTION2'\n"
    best = rank_candidates([a, b], scene, model,
                           TransitionModel(), NoveltyModel(), {})
    assert best == b
