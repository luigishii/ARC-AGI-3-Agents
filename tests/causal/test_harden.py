import json

from arcengine import GameAction, GameState

from agents.causal.llm import parse_goal, client_kind, NullLLMClient
from agents.causal.agent import CausalObjectAgent


# --- item 1: parsing robusto ---
def test_parse_goal_strips_markdown_fence():
    txt = '```json\n{"type":"press","action":"ACTION1"}\n```'
    assert parse_goal(txt) == {"type": "press", "action": "ACTION1"}


def test_parse_goal_handles_prose_around_json():
    txt = 'Sure! Here is the goal:\n{"type":"press","action":"ACTION3"} — hope it helps'
    assert parse_goal(txt) == {"type": "press", "action": "ACTION3"}


def test_parse_goal_balances_first_object():
    txt = '{"type":"click_cell","gx":1,"gy":2} {"type":"press","action":"X"}'
    assert parse_goal(txt) == {"type": "click_cell", "gx": 1, "gy": 2}


def test_parse_goal_garbage_is_none():
    assert parse_goal("no json here") is None


# --- item 2: client_kind ---
def test_client_kind_null():
    assert client_kind(NullLLMClient()) == "null"


# --- harness p/ o agente ---
class _Fake:
    def __init__(self, canned):
        self.canned = canned
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.canned


class _SeqFake:
    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0

    def complete(self, prompt):
        r = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return r


class _Frame:
    def __init__(self, frame):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = 0
        self.available_actions = [GameAction.ACTION1]
        self.full_reset = False


def _grid(v=3):
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = v
    return [g]


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_llm_kind_recorded(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    assert a._llm_kind == "null"       # sem QWEN_MODEL_PATH → NullLLMClient


# --- item 5: guardrail de orçamento ---
def test_budget_zero_blocks_llm(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_LLM_MAX_CALLS="0")
    fake = _Fake('{"type":"press","action":"ACTION1"}')
    a._llm = fake
    a.choose_action([], _Frame(_grid()))
    assert fake.calls == 0             # orçamento 0 → não consulta
    assert a._goal is None


# --- item 3: self-repair (erro → re-pergunta → aceita) ---
_GOOD = "def transition(obj, action, ctx):\n    o = dict(obj)\n    o['x'] = o['x'] + 1\n    return o\n"
_BAD = "def transition(obj, action, ctx):\n    o = dict(obj)\n    o['x'] = o['x'] + 2\n    return o\n"
_BUF = [
    {"before": {"x": 1, "y": 0}, "action": "A", "context": {}, "after": {"x": 2, "y": 0}},
    {"before": {"x": 5, "y": 1}, "action": "A", "context": {}, "after": {"x": 6, "y": 1}},
    {"before": {"x": 0, "y": 2}, "action": "A", "context": {}, "after": {"x": 1, "y": 2}},
]


def test_self_repair_recovers_after_bad(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_TYPED="1", CAUSAL_REPAIR="1")
    a._llm = _SeqFake([json.dumps({"type": "code", "source": _BAD}),
                       json.dumps({"type": "code", "source": _GOOD})])
    a._type_buffer["shp"] = list(_BUF)
    assert a._try_learn_type_rule("shp") is True     # 1ª falha → reparo → aceita
    assert "shp" in a._typed.sources


def test_no_repair_budget_gives_up(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_TYPED="1", CAUSAL_REPAIR="0")
    a._llm = _SeqFake([json.dumps({"type": "code", "source": _BAD}),
                       json.dumps({"type": "code", "source": _GOOD})])
    a._type_buffer["shp"] = list(_BUF)
    assert a._try_learn_type_rule("shp") is False    # sem reparo → desiste na 1ª ruim
