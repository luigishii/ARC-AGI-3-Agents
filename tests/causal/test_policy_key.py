from arcengine import GameAction
from agents.causal.policy import Candidate, action_key


def test_candidate_has_six_fields():
    c = Candidate(GameAction.ACTION1, None, None, "ACTION1", False)
    assert c.has_object is False
    assert c._fields == ("action", "x", "y", "key", "has_object", "obj_size")


def test_action_key_simple_is_name():
    assert action_key(GameAction.ACTION1) == "ACTION1"


def test_action_key_complex_same_cell_same_key():
    a = GameAction.ACTION6
    assert action_key(a, (2, 3)) == "ACTION6@cell=2,3"
    assert action_key(a, (2, 3)) == action_key(a, (2, 3))


def test_action_key_complex_diff_cell_diff_key():
    a = GameAction.ACTION6
    assert action_key(a, (2, 3)) != action_key(a, (3, 2))


def test_action_key_complex_none_cell_is_empty():
    assert action_key(GameAction.ACTION6, None) == "ACTION6@empty"
