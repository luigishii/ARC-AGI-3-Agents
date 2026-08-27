import json

import numpy as np

from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects
from agents.causal.policy import Candidate


class _Fake:
    def __init__(self, canned):
        self.canned = canned
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.canned


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def _grid_at(col):
    g = np.zeros((8, 8), dtype=int)
    g[1, col] = 3
    return g


# --- close-loop popula tabela de efeitos (η) + buffer por tipo ---
def test_observe_types_populates(monkeypatch):
    a = _agent(monkeypatch)
    s1 = match_objects(None, parse(_grid_at(1)))
    s2 = match_objects(s1, parse(_grid_at(2)))     # objeto move 1 coluna
    a._observe_types(s1, s2, "ACTION1")
    assert a._type_buffer                            # algum tipo bufferizado
    tau = next(iter(a._type_buffer))
    tr = a._type_buffer[tau][0]
    assert tr["before"]["x"] == 1 and tr["after"]["x"] == 2
    assert a._etable.rows                            # linha (τ,ACTION1,u) registrada


# --- síntese fatorada f_τ validada por accept_rule ---
_MOVE_BUF = [
    {"before": {"x": 1, "y": 0}, "action": "A", "context": {}, "after": {"x": 2, "y": 0}},
    {"before": {"x": 5, "y": 1}, "action": "A", "context": {}, "after": {"x": 6, "y": 1}},
    {"before": {"x": 0, "y": 2}, "action": "A", "context": {}, "after": {"x": 1, "y": 2}},
]
_GOOD = "def transition(obj, action, ctx):\n    o = dict(obj)\n    o['x'] = o['x'] + 1\n    return o\n"
_BAD = "def transition(obj, action, ctx):\n    o = dict(obj)\n    o['x'] = o['x'] + 2\n    return o\n"


def test_try_learn_type_rule_accepts_valid(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_TYPED="1")
    a._llm = _Fake(json.dumps({"type": "code", "source": _GOOD}))
    a._type_buffer["shp"] = list(_MOVE_BUF)
    assert a._try_learn_type_rule("shp") is True
    assert "shp" in a._typed.sources


def test_try_learn_type_rule_rejects_invalid(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_TYPED="1")
    a._llm = _Fake(json.dumps({"type": "code", "source": _BAD}))
    a._type_buffer["shp"] = list(_MOVE_BUF)
    assert a._try_learn_type_rule("shp") is False
    assert "shp" not in a._typed.sources


def test_try_learn_needs_min_obs(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_TYPED="1")
    a._llm = _Fake(json.dumps({"type": "code", "source": _GOOD}))
    a._type_buffer["shp"] = _MOVE_BUF[:1]          # poucas transições
    assert a._try_learn_type_rule("shp") is False


# --- η-explore escolhe a ação de linha mais ambígua ---
def test_eta_explore_picks_ambiguous(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_ETA="1")
    for _ in range(6):
        a._etable.observe("t", "ACTION1", "", "x")     # determinística
    a._etable.observe("t", "ACTION2", "", "x")         # misturada
    a._etable.observe("t", "ACTION2", "", "y")
    cands = [Candidate(None, None, None, "ACTION1", False),
             Candidate(None, None, None, "ACTION2", False)]
    assert a._eta_explore(cands) == "ACTION2"


# --- guardas: default off não sintetiza tipo nem consulta typed ---
def test_typed_off_by_default(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")            # TYPED não setado → off
    a._llm = _Fake(json.dumps({"type": "code", "source": _GOOD}))
    a._type_buffer["shp"] = list(_MOVE_BUF)
    assert a._typed_on is False
    # o gatilho só roda sob CAUSAL_TYPED; aqui o modelo continua vazio
    assert a._typed.sources == {}
