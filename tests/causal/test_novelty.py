from agents.causal.perception import Scene, Object
from agents.causal.novelty import state_signature


def _obj(cells, color=3):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return Object(color, cset, bbox, centroid, len(cells), "h")


def test_empty_scene_signature_is_empty_string():
    assert state_signature(Scene(objects=[], grid=None)) == ""


def test_same_config_same_signature():
    a = Scene(objects=[_obj([(5, 5)])], grid=None)
    b = Scene(objects=[_obj([(5, 5)])], grid=None)
    assert state_signature(a) == state_signature(b)


def test_object_moved_to_other_cell_changes_signature():
    a = Scene(objects=[_obj([(5, 5)])], grid=None)      # célula (0,0)
    b = Scene(objects=[_obj([(50, 50)])], grid=None)    # célula (4,4)
    assert state_signature(a) != state_signature(b)


def test_signature_order_independent():
    o1 = _obj([(5, 5)], color=3)
    o2 = _obj([(50, 50)], color=4)
    a = Scene(objects=[o1, o2], grid=None)
    b = Scene(objects=[o2, o1], grid=None)
    assert state_signature(a) == state_signature(b)
