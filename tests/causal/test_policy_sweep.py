from arcengine import GameAction
from agents.causal.perception import Scene, Object
from agents.causal.causal_model import CausalModel
from agents.causal.policy import Policy, candidates, cell_of


def _obj(cells, color=3):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return Object(color, cset, bbox, centroid, len(cells), "h")


def test_object_cell_preferred_over_empty_when_unexplored():
    # objeto em (row=50,col=50) → célula (4,4), NÃO a primeira da ordem de
    # geração (0,0). Sem o bônus, o empate entre células inexploradas cairia
    # em (0,0); com o bônus +0.5, a policy escolhe a célula ocupada (4,4).
    scene = Scene(objects=[_obj([(50, 50)])], grid=None)
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    chosen = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0)
    assert cell_of(chosen.x, chosen.y) == (4, 4)          # bônus +0.5 vence


def test_sweep_avoids_known_none_cell():
    scene = Scene(objects=[], grid=None)
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    first = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0)
    # registra a célula escolhida como 'none' (empty→empty gera Effect none)
    empty = Scene(objects=[], grid=None)
    model.observe(empty, first.key, empty)
    second = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0)
    assert second.key != first.key                        # não repete célula morta


def test_reproducible_cell_key_is_stable_and_predictable():
    scene = Scene(objects=[], grid=None)
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    c = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0)
    # cena→cena seguinte com mudança real sob a mesma chave de célula
    before = Scene(objects=[_obj([(5, 5)])], grid=None)
    after = Scene(objects=[_obj([(6, 6)], color=4)], grid=None)
    eff = model.observe(before, c.key, after)
    assert eff.kind != "none"
    pred, conf = model.predict(c.key)
    assert pred is not None and pred.kind == eff.kind
