from agents.causal.agent import CausalObjectAgent, REWARD_DEFER_MAX


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_defer_until_deadline_when_no_avatar(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    a._reward_fn = None                      # sem avatar aprendido
    for _ in range(REWARD_DEFER_MAX):
        assert a._should_learn_reward() is False
    assert a._should_learn_reward() is True   # deadline atingido


def test_learn_immediately_when_avatar_known(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    a._reward_fn = None
    a._move.avatar_counts = {7: 3}            # avatar conhecido
    assert a._should_learn_reward() is True


def test_no_relearn_when_reward_exists(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    a._reward_fn = lambda s: (0.0, False)     # já aprendida
    assert a._should_learn_reward() is False
