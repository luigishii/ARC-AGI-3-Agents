import math

from agents.causal.goals import value_fn_from_reward


def test_value_reads_scalar_from_tuple():
    vf = value_fn_from_reward(lambda state: (42.0, False))
    assert vf(["anything"]) == 42.0


def test_value_reads_bare_scalar():
    vf = value_fn_from_reward(lambda state: 7)
    assert vf(["anything"]) == 7.0


def test_value_exception_is_neg_inf():
    def boom(state):
        raise ValueError("boom")
    vf = value_fn_from_reward(boom)
    assert vf(["x"]) == float("-inf")
    assert math.isinf(vf(["x"]))
