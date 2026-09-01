from agents.causal.goals import grounded_multi_reward_fn


def test_multi_reward_sums_small_samecolor_excludes_big():
    rf = grounded_multi_reward_fn(max_size=64)
    # dois marcadores cor 11 pequenos (dist 2) + barra HUD cor 7 enorme (ignorada)
    state = [("h", {"x": 0, "y": 0, "color": 11, "size": 9}),
             ("h", {"x": 2, "y": 0, "color": 11, "size": 9}),
             ("h", {"x": 30, "y": 0, "color": 7, "size": 999})]  # HUD grande -> excluida
    assert rf(state) == (-2.0, False)


def test_multi_reward_goal_when_all_aligned():
    rf = grounded_multi_reward_fn()
    state = [("h", {"x": 5, "y": 5, "color": 11, "size": 9}),
             ("h", {"x": 5, "y": 5, "color": 11, "size": 9})]
    assert rf(state) == (0.0, True)


def test_multi_reward_safe_on_garbage():
    assert grounded_multi_reward_fn()("x") == (0.0, False)
