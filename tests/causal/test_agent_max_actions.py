from agents.causal.agent import UNKNOWN_BUDGET, CausalObjectAgent


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a._cleanup = False
    a._init_causal_state()
    return a


def test_default_max_actions_is_unknown_budget(monkeypatch):
    # Era 80 (e o early-exit cortava na acao 52). Jogo nao-visto e exatamente o caso
    # da eval privada; medido nos 25 jogos, o teto apertado custava metade do score.
    monkeypatch.delenv("CAUSAL_MAX_ACTIONS", raising=False)
    a = _agent()
    assert a.MAX_ACTIONS == UNKNOWN_BUDGET


def test_env_overrides_max_actions(monkeypatch):
    monkeypatch.setenv("CAUSAL_MAX_ACTIONS", "5")
    a = _agent()
    assert a.MAX_ACTIONS == 5
