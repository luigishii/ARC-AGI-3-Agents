from agents.causal.goals import grounded_pattern_reward_fn


def test_pattern_reward_zero_when_halves_match():
    rf = grounded_pattern_reward_fn(split=32)
    # topo (referencia): cores 9,14 ; baixo (editavel): cores 9,14 -> casam -> reward 0, goal
    state = [("h", {"x": 5, "y": 5, "color": 9, "size": 4}),
             ("h", {"x": 9, "y": 5, "color": 14, "size": 4}),
             ("h", {"x": 5, "y": 50, "color": 9, "size": 4}),
             ("h", {"x": 9, "y": 50, "color": 14, "size": 4})]
    assert rf(state) == (0.0, True)


def test_pattern_reward_penalizes_mismatch():
    rf = grounded_pattern_reward_fn(split=32)
    # baixo tem cor 15 onde deveria 14 -> 2 de diferenca simetrica (falta 14, sobra 15)
    state = [("h", {"x": 5, "y": 5, "color": 9, "size": 4}),
             ("h", {"x": 9, "y": 5, "color": 14, "size": 4}),
             ("h", {"x": 5, "y": 50, "color": 9, "size": 4}),
             ("h", {"x": 9, "y": 50, "color": 15, "size": 4})]
    r, g = rf(state)
    assert r == -2.0 and g is False


def test_pattern_reward_safe_and_ignores_big():
    rf = grounded_pattern_reward_fn()
    assert rf("x") == (0.0, False)
    # so um bloco HUD enorme -> nada a casar -> nao dispara goal
    assert rf([("h", {"x": 0, "y": 0, "color": 7, "size": 999})]) == (0.0, False)
