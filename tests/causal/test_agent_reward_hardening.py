import numpy as np

from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects


class _Seq:
    """FakeLLM que devolve respostas canned em sequência (1 por chamada)."""
    def __init__(self, canned):
        self.canned = list(canned)
        self.calls = 0

    def complete(self, prompt):
        r = self.canned[min(self.calls, len(self.canned) - 1)]
        self.calls += 1
        return r


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def _scene(n):
    g = np.zeros((8, 8), dtype=int)
    for i in range(n):
        g[0, i * 2] = 3
    return match_objects(None, parse(g))


def _fill_buffer(a):
    for n in (1, 2, 3):                       # 3 cenas distintas → gradiente julgável
        a._buffer.append((_scene(n), "ACTION1", "structural"))


def _reward_json(body):
    import json
    return json.dumps({"type": "code",
                       "source": "def reward_function(state):\n    " + body})


# --- reward constante é rejeitada; _reward_rejected sobe; _reward_fn fica None ---
def test_rejects_constant_reward(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    _fill_buffer(a)
    # usa o state (passa o check estático) mas é CONSTANTE -> barrada pelo comportamental
    a._llm = _Seq([_reward_json("_ = len(state)\n    return (0, False)")])
    ok = a._try_learn_reward(_scene(2))
    assert ok is False
    assert a._reward_fn is None
    assert a._reward_rejected >= 1


# --- reward graduada é aceita ---
def test_accepts_graded_reward(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    _fill_buffer(a)
    a._llm = _Seq([_reward_json("return (len(state), False)")])
    ok = a._try_learn_reward(_scene(2))
    assert ok is True
    assert a._reward_fn is not None


# --- self-repair: 1ª binária (rejeitada) -> 2ª graduada (aceita) ---
def test_self_repair_recovers(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="1")
    _fill_buffer(a)
    a._llm = _Seq([_reward_json("_ = len(state)\n    return (0, False)"),  # rodada 1: rejeitada
                   _reward_json("return (len(state), False)")])            # rodada 2: aceita
    ok = a._try_learn_reward(_scene(2))
    assert ok is True
    assert a._reward_rejected >= 1


# --- prompt estrito tem as instruções-chave ---
def test_prompt_is_strict(monkeypatch):
    a = _agent(monkeypatch)
    p = a._build_reward_prompt(_scene(2))
    assert "GRADUADO" in p
    assert "magic" in p.lower() or "hardcode" in p.lower()
    assert "goal_flag=True" in p


# --- phase2_stats expõe reward_rejected ---
def test_phase2_has_reward_rejected(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_rejected = 4
    assert a.phase2_stats()["reward_rejected"] == 4
