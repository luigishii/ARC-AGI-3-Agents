from arcengine import GameAction, GameState
from agents.causal.agent import CausalObjectAgent


class _ListFake:
    """complete_many devolve uma lista canned; complete devolve o 1º."""
    def __init__(self, items):
        self.items = items
        self.many_calls = 0

    def complete(self, prompt):
        return self.items[0]

    def complete_many(self, prompt, n):
        self.many_calls += 1
        return list(self.items)


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


def _agent(monkeypatch, samples="3"):
    monkeypatch.setenv("CAUSAL_LLM", "1")
    monkeypatch.setenv("CAUSAL_SAMPLES", samples)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._cleanup = False
    a._init_causal_state()
    return a


def test_samples_and_sets_code_goal(monkeypatch):
    a = _agent(monkeypatch, samples="3")
    cands = [
        '{"type":"code","source":"def decide(scene): return \'ACTION1\'"}',
        '{"type":"code","source":"def decide(scene): return \'ACTION1\'"}',
        'lixo',
    ]
    fake = _ListFake(cands)
    a._llm = fake
    a.choose_action([], _Frame(_grid(3), available=[GameAction.ACTION1]))
    assert fake.many_calls == 1                      # amostrou em lote
    assert a._goal is not None and a._goal["type"] == "code"


def test_single_sample_no_batch(monkeypatch):
    a = _agent(monkeypatch, samples="1")
    fake = _ListFake(['{"type":"press","action":"ACTION1"}'])
    a._llm = fake
    a.choose_action([], _Frame(_grid(3), available=[GameAction.ACTION1]))
    assert fake.many_calls == 0                      # n=1 → não usa complete_many
    assert a._goal == {"type": "press", "action": "ACTION1"}
