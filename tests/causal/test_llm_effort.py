from agents.causal.llm import resolve_effort, NullLLMClient


def test_resolve_effort_defaults(monkeypatch):
    monkeypatch.delenv("CAUSAL_EFFORT", raising=False)
    monkeypatch.delenv("CAUSAL_DIRECT_EFFORT", raising=False)
    assert resolve_effort(None) == "medium"
    assert resolve_effort("low") == "low"           # explicito vence


def test_resolve_effort_env(monkeypatch):
    monkeypatch.setenv("CAUSAL_EFFORT", "high")
    assert resolve_effort(None) == "high"


def test_null_client_accepts_effort_kwarg():
    assert NullLLMClient().complete("x", effort="low") == ""
