from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent


class _Fake:
    def __init__(self, canned):
        self.canned = canned
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.canned


class _Frame:
    def __init__(self, frame, available=None, levels=0):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = levels
        self.available_actions = available or [GameAction.ACTION1]
        self.full_reset = False


def _grid(v=3):
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = v
    return [g]


def _agent(monkeypatch, **env):
    env.setdefault("CAUSAL_LLM", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_direct_uses_valid_action(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    act = a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert act.name == "ACTION1"
    assert a.phase2_stats()["direct_hits"] == 1
    assert a.phase2_stats()["direct_calls"] == 1


def test_direct_invalid_falls_through(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1")
    a._llm = _Fake('{"type":"press","action":"ACTION5"}')   # fora do available
    act = a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert act is not None                                   # nao crasha
    assert a.phase2_stats()["direct_hits"] == 0
    assert a.phase2_stats()["direct_calls"] == 1


def test_direct_off_never_queries(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="0")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert a.phase2_stats()["direct_calls"] == 0


def test_direct_cooldown(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1", CAUSAL_DIRECT_COOLDOWN="2")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    a.choose_action([], _Frame(_grid(3), available=[GameAction.ACTION1]))   # consulta
    a.choose_action([], _Frame(_grid(4), available=[GameAction.ACTION1]))   # cooldown -> nao
    assert a.phase2_stats()["direct_calls"] == 1


def test_direct_budget_exhausted(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1", CAUSAL_LLM_MAX_CALLS="0")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert a.phase2_stats()["direct_calls"] == 0


def test_direct_skips_persistent_goal(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert a._goal is None     # bloco de meta-persistente pulado sob direct


def test_direct_click_cell(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1")
    a._llm = _Fake('{"type":"click_cell","gx":2,"gy":3}')
    act = a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION6]))
    assert act is not None
    assert a.phase2_stats()["direct_hits"] == 1
