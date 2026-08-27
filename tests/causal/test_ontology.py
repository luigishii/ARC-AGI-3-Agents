import math
from dataclasses import replace

from agents.causal.perception import Object
from agents.causal.ontology import (
    effect_signature, normalized_entropy, LocalEffectTable, ontology_error,
)


def _obj(color=3, cells=((0, 0),), oid=1):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    from agents.causal.perception import _shape_hash
    bbox = (min(rs), min(cs), max(rs), max(cs))
    return replace(Object(color, cset, bbox, centroid, len(cells), _shape_hash(cset)), id=oid)


# --- effect_signature(Δ): assinatura categórica, deltas exatos descartados ---
def test_signature_no_change():
    o = _obj()
    assert effect_signature(o, o) == "no_change"


def test_signature_born_and_gone():
    o = _obj()
    assert effect_signature(None, o) == "born"
    assert effect_signature(o, None) == "gone"


def test_signature_move_column_is_x():
    prev = _obj(cells=((5, 5),))
    curr = _obj(cells=((5, 8),))          # mesma linha, coluna muda
    assert effect_signature(prev, curr) == "x"


def test_signature_move_both_is_xy_and_discards_exact_delta():
    prev = _obj(cells=((5, 5),))
    curr_small = _obj(cells=((7, 8),))    # +2, +3
    curr_big = _obj(cells=((40, 60),))    # deltas diferentes
    assert effect_signature(prev, curr_small) == "x,y"
    assert effect_signature(prev, curr_small) == effect_signature(prev, curr_big)


def test_signature_recolor():
    prev = _obj(color=3, cells=((5, 5),))
    curr = _obj(color=7, cells=((5, 5),))
    assert effect_signature(prev, curr) == "recolor"


def test_signature_pixels_on_shape_change():
    prev = _obj(cells=((5, 5),))
    curr = _obj(cells=((5, 5), (5, 6)))   # cresceu → shape muda
    assert "pixels" in effect_signature(prev, curr)


# --- normalized_entropy ---
def test_normalized_entropy_uniform_is_one():
    assert abs(normalized_entropy([0.25, 0.25, 0.25, 0.25]) - 1.0) < 1e-9


def test_normalized_entropy_degenerate_is_zero():
    assert normalized_entropy([1.0, 0.0, 0.0]) == 0.0
    assert normalized_entropy([1.0]) == 0.0


# --- LocalEffectTable: Dirichlet posterior + entropia por linha ---
def test_table_posterior_sums_to_one():
    t = LocalEffectTable(alpha=1.0)
    t.observe("t", "A", "u", "x")
    t.observe("t", "A", "u", "x")
    t.observe("t", "A", "u", "y")
    post = t.posterior(("t", "A", "u"))
    assert abs(sum(post.values()) - 1.0) < 1e-9
    assert post["x"] > post["y"]          # mais observado → maior posterior


def test_table_deterministic_row_has_low_entropy():
    t = LocalEffectTable(alpha=0.1)
    for _ in range(50):
        t.observe("t", "A", "u", "x")     # linha determinística
    t.observe("t", "B", "u", "x")         # cria alfabeto>1 p/ normalizar
    t.observe("t", "B", "u", "y")
    det = t.row_entropy(("t", "A", "u"))
    mixed = t.row_entropy(("t", "B", "u"))
    assert det < mixed
    assert det < 0.2


def test_table_serialization_roundtrip():
    t = LocalEffectTable(alpha=0.5)
    t.observe("t", "A", "u", "x")
    t2 = LocalEffectTable.from_dict(t.to_dict())
    assert t2.posterior(("t", "A", "u")) == t.posterior(("t", "A", "u"))


# --- ontology_error(η): noisy-OR de duas entropias normalizadas ---
def test_ontology_error_noisy_or():
    # noisy-OR = 1 - (1-a)(1-b)
    assert abs(ontology_error(0.0, 0.0) - 0.0) < 1e-9
    assert abs(ontology_error(1.0, 0.0) - 1.0) < 1e-9
    assert abs(ontology_error(0.5, 0.5) - 0.75) < 1e-9


def test_ontology_error_monotonic():
    assert ontology_error(0.2, 0.2) < ontology_error(0.8, 0.2)
