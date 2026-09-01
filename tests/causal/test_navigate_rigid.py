from types import SimpleNamespace as NS

from agents.causal.navigate import _moved_object, MovementModel


def _o(oid, centroid, shape, size):
    return NS(id=oid, centroid=centroid, shape_hash=shape, size=size)


def _scene(objs):
    return NS(objects=objs)


def test_isolates_rigid_ignoring_shrinking_bar():
    prev = _scene([_o(1, (5, 5), "A", 9), _o(2, (0, 5), "BAR", 10)])
    curr = _scene([_o(1, (5, 10), "A", 9),    # avatar transladou rigido
                   _o(2, (0, 3), "bar2", 7)])  # barra encolheu+shiftou (nao-rigido)
    m = _moved_object(prev, curr)
    assert m is not None and m[0] == 1


def test_two_rigid_movers_is_none():
    prev = _scene([_o(1, (5, 5), "A", 9), _o(2, (0, 0), "B", 4)])
    curr = _scene([_o(1, (5, 10), "A", 9), _o(2, (0, 5), "B", 4)])
    assert _moved_object(prev, curr) is None


def test_zero_movers_is_none():
    prev = _scene([_o(1, (5, 5), "A", 9)])
    curr = _scene([_o(1, (5, 5), "A", 9)])
    assert _moved_object(prev, curr) is None


def test_single_rigid_mover_returned():
    prev = _scene([_o(1, (5, 5), "A", 9)])
    curr = _scene([_o(1, (7, 5), "A", 9)])
    m = _moved_object(prev, curr)
    assert m == (1, (2, 0))


def test_movement_model_learns_avatar_under_hud_noise():
    prev = _scene([_o(1, (5, 5), "A", 9), _o(2, (0, 5), "BAR", 10)])
    curr = _scene([_o(1, (5, 10), "A", 9), _o(2, (0, 3), "bar2", 7)])
    mm = MovementModel()
    mm.observe("ACTION1", prev, curr)
    assert mm.avatar_id() == 1
