from agents.causal.agent import CausalObjectAgent


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a._cleanup = False
    a._init_causal_state()
    return a


def test_default_max_actions_is_80(monkeypatch):
    monkeypatch.delenv("CAUSAL_MAX_ACTIONS", raising=False)
    a = _agent()
    assert a.MAX_ACTIONS == 80


def test_env_overrides_max_actions(monkeypatch):
    monkeypatch.setenv("CAUSAL_MAX_ACTIONS", "5")
    a = _agent()
    assert a.MAX_ACTIONS == 5
