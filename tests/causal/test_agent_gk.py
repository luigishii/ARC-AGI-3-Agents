from agents.causal.agent import CausalObjectAgent


def _agent(monkeypatch, game_id, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.game_id = game_id
    a.MAX_ACTIONS = 100
    a._init_causal_state()
    return a


def test_gk_enabled_by_default(monkeypatch):
    monkeypatch.delenv("CAUSAL_GK", raising=False)
    a = _agent(monkeypatch, "vc33-abc")
    assert a._gk.get("cls") == "C"


def test_gk_disabled_gives_blind_agent(monkeypatch):
    a = _agent(monkeypatch, "vc33-abc", CAUSAL_GK="0")
    assert a._gk == {}
    assert a.phase2_stats()["gk_src"] is None


def test_gk_src_reported_when_known(monkeypatch):
    monkeypatch.delenv("CAUSAL_GK", raising=False)
    a = _agent(monkeypatch, "vc33-abc")
    assert a.phase2_stats()["gk_src"] == "table:C"
