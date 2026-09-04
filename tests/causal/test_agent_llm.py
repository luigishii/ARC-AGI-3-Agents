from arcengine import GameAction, GameState
from agents.causal.agent import CausalObjectAgent


class _CountingFake:
    def __init__(self, canned):
        self.canned = canned
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.canned


class _Frame:
    def __init__(self, frame, state=GameState.NOT_FINISHED, levels=0, available=None):
        self.frame = frame
        self.state = state
        self.levels_completed = levels
        self.available_actions = available or [GameAction.ACTION1]
        self.full_reset = False


def _grid(v):
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = v
    return [g]


def _agent(monkeypatch, llm="1"):
    monkeypatch.setenv("CAUSAL_LLM", llm)
    monkeypatch.setenv("CAUSAL_LLM_DEFER", "0")
    monkeypatch.setenv("CAUSAL_MAX_ACTIONS", "10000")
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 10     # pula sweep de tipos (< len(avail)*2)
    a.game_id = "test"
    a.MAX_ACTIONS = 10000
    a._cleanup = False
    a._init_causal_state()
    return a


def test_uses_llm_goal(monkeypatch):
    a = _agent(monkeypatch, llm="1")
    a._llm = _CountingFake('{"type":"press","action":"ACTION1"}')
    act = a.choose_action([], _Frame(_grid(3), available=[GameAction.ACTION1]))
    assert act.name == "ACTION1"          # usou a meta press do LLM neste passo
    assert a._goal is None                # press e single-shot -> limpa apos usar


def test_invalid_response_falls_back(monkeypatch):
    a = _agent(monkeypatch, llm="1")
    a._llm = _CountingFake("lixo sem json")
    act = a.choose_action([], _Frame(_grid(3)))
    assert a._goal is None
    assert act is not None


def test_off_never_queries(monkeypatch):
    a = _agent(monkeypatch, llm="0")
    fake = _CountingFake('{"type":"press","action":"ACTION1"}')
    a._llm = fake
    a.choose_action([], _Frame(_grid(3)))
    assert fake.calls == 0
    assert a._goal is None


def test_sparse_no_requery_with_active_goal(monkeypatch):
    a = _agent(monkeypatch, llm="1")
    fake = _CountingFake('{"type":"press","action":"ACTION1"}')
    a._llm = fake
    for _ in range(5):
        a.choose_action([], _Frame(_grid(3), available=[GameAction.ACTION1]))
    assert fake.calls == 1     # meta ativa persiste → 1 chamada só


def test_levelup_clears_goal(monkeypatch):
    a = _agent(monkeypatch, llm="1")
    # meta 'code' persiste (nao e single-shot como press) -> testa que o LEVEL-UP a limpa
    a._llm = _CountingFake('{"type":"code","source":"def decide(scene):\\n    return \'ACTION1\'\\n"}')
    a.choose_action([], _Frame(_grid(3), levels=0, available=[GameAction.ACTION1]))
    assert a._goal is not None
    a.choose_action([], _Frame(_grid(4), levels=1, available=[GameAction.ACTION1]))
    assert a._goal is None
