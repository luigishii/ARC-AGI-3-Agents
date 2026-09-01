from agents.causal.goals import grounded_reward_fn, value_fn_from_reward


def test_grounded_reward_negative_distance_and_goal():
    rf = grounded_reward_fn(9, 5)   # avatar cor 9, alvo cor 5
    state = [("h", {"x": 0, "y": 0, "color": 9}), ("h", {"x": 3, "y": 4, "color": 5})]
    r, g = rf(state)
    assert r == -7.0 and g is False
    # avatar em cima do alvo -> reward 0, goal True
    state2 = [("h", {"x": 2, "y": 2, "color": 9}), ("h", {"x": 2, "y": 2, "color": 5})]
    assert rf(state2) == (0.0, True)


def test_grounded_reward_missing_object_is_safe():
    rf = grounded_reward_fn(9, 5)
    assert rf([("h", {"x": 0, "y": 0, "color": 9})]) == (0.0, False)  # sem alvo cor 5
    assert rf("garbage") == (0.0, False)                             # entrada invalida


def test_grounded_reward_plugs_into_value_fn():
    rf = grounded_reward_fn(9, 5)
    vf = value_fn_from_reward(rf)   # o IW/rprog usam o escalar
    near = [("h", {"x": 2, "y": 2, "color": 9}), ("h", {"x": 3, "y": 2, "color": 5})]
    far = [("h", {"x": 2, "y": 2, "color": 9}), ("h", {"x": 40, "y": 2, "color": 5})]
    assert vf(near) > vf(far)       # mais perto = valor maior (menos negativo)
