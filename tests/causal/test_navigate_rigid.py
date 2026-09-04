from types import SimpleNamespace as NS

from agents.causal.navigate import _moved_object, MovementModel


def _o(oid, centroid, shape, size, color=None):
    return NS(id=oid, centroid=centroid, shape_hash=shape, size=size,
              color=color if color is not None else oid)


def _scene(objs):
    return NS(objects=objs)


def test_isolates_rigid_ignoring_shrinking_bar():
    prev = _scene([_o(1, (5, 5), "A", 9), _o(2, (0, 5), "BAR", 10)])
    curr = _scene([_o(1, (5, 10), "A", 9),    # avatar transladou rigido
                   _o(2, (0, 3), "bar2", 7)])  # barra encolheu+shiftou (nao-rigido)
    m = _moved_object(prev, curr)
    assert m is not None and m[0] == 1


def test_two_rigid_movers_different_vectors_is_none():
    prev = _scene([_o(1, (5, 5), "A", 9), _o(2, (0, 0), "B", 4)])
    curr = _scene([_o(1, (5, 10), "A", 9), _o(2, (3, 0), "B", 4)])   # vetores distintos: ambiguo
    assert _moved_object(prev, curr) is None


def test_composite_avatar_same_vector_is_one_mover():
    """ls20: avatar bicolor (cor 12 size 10 + cor 9 size 15) translada como UMA entidade.
    Movers rigidos com o MESMO vetor = avatar composto -> devolve a maior parte + vetor."""
    prev = _scene([_o(1, (5, 5), "A", 10), _o(2, (5, 7), "B", 15), _o(3, (0, 5), "BAR", 10)])
    curr = _scene([_o(1, (5, 10), "A", 10), _o(2, (5, 12), "B", 15), _o(3, (0, 3), "bar2", 7)])
    assert _moved_object(prev, curr) == (2, (0, 5))


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


def test_movement_model_tracks_companions_and_navigate_ignores_them():
    """A outra metade do avatar composto NAO pode virar alvo (ls20: mirava a propria
    parte cor 12 a 3px e 'alcancava' na hora)."""
    from agents.causal.navigate import navigate
    mm = MovementModel()
    prev = _scene([_o(1, (5, 5), "A", 10), _o(2, (5, 7), "B", 15), _o(9, (40, 40), "T", 4)])
    curr = _scene([_o(1, (5, 10), "A", 10), _o(2, (5, 12), "B", 15), _o(9, (40, 40), "T", 4)])
    mm.observe("ACTION3", prev, curr)
    assert mm.avatar_id() == 2
    assert mm.avatar_parts() == {1, 2}
    # alvo: o unico objeto que nao e parte do avatar; direita (ACTION3 = (0,+5)) aproxima
    # em coluna mas o alvo esta em (40,40): abaixo+direita. Sem move p/ baixo, escolhe ACTION3.
    assert navigate(curr, mm) == "ACTION3"
    # sem exclusao a parte 1 (a 3px) seria o alvo "raro" e navigate devolveria None/reached
