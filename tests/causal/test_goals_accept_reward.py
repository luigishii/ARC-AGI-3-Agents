from agents.causal.goals import accept_reward

_ST3 = [[("h", {"color": 1, "size": 1})],
        [("h", {"color": 1, "size": 1}), ("h", {"color": 2, "size": 2})],
        [("h", {"color": 1, "size": 1}), ("h", {"color": 2, "size": 2}),
         ("h", {"color": 3, "size": 3})]]   # 3 estados distintos (len 1,2,3)


def test_reject_always_true():
    src = "def reward_function(state):\n    return (1, True)"
    ok, reason = accept_reward(src, _ST3)
    assert ok is False and "falso-positivo" in reason


def test_reject_constant_scalar():
    src = "def reward_function(state):\n    return (0, False)"   # constante em estados distintos
    ok, reason = accept_reward(src, _ST3)
    assert ok is False and "CONSTANTE" in reason


def test_accept_graded():
    src = "def reward_function(state):\n    return (len(state), False)"   # varia 1,2,3
    ok, reason = accept_reward(src, _ST3)
    assert ok is True


def test_reject_raises_on_real_state():
    src = "def reward_function(state):\n    return (1 / (len(state) - 1), False)"  # ZeroDiv em len==1
    ok, reason = accept_reward(src, _ST3)
    assert ok is False and "exceção" in reason


def test_reject_non_compiling():
    ok, reason = accept_reward("not python at all", _ST3)
    assert ok is False and "compila" in reason


def test_cold_start_accepts_few_states():
    src = "def reward_function(state):\n    return (0, False)"
    ok, reason = accept_reward(src, _ST3[:1])       # 1 estado < min_states
    assert ok is True


def test_identical_states_skip_gradient():
    same = [[("h", {"color": 1, "size": 1})]] * 3   # 3 estados IDÊNTICOS
    src = "def reward_function(state):\n    return (5, False)"   # constante, mas estados iguais
    ok, reason = accept_reward(src, same)
    assert ok is True                                # pula o teste de gradiente
