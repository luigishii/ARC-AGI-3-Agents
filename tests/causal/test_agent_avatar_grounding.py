import numpy as np

from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects


def _scene(coords):
    g = np.zeros((16, 16), dtype=int)
    for (r, c) in coords:
        g[r, c] = 3
    return match_objects(None, parse(g))


class _Seq:
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


def test_prompt_uses_learned_avatar_index(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (0, 8)])
    objs = list(scene.objects)
    a._move.avatar_counts = {objs[1].id: 5}     # forca o avatar a ser o indice 1
    p = a._build_reward_prompt(scene)
    assert "state[1]" in p
    assert "pts[1]" in p
    assert "OBJETO CONTROLAVEL" in p


def test_prompt_fallback_when_no_avatar(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (0, 8)])
    p = a._build_reward_prompt(scene)                # avatar_counts vazio -> None
    assert "OBJETO CONTROLAVEL" not in p
    assert "a=pts[0]" in p


def test_grounded_distance_reward_accepted(monkeypatch):
    import json
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    for coords in ([(0, 0), (0, 8)], [(0, 0), (0, 4)], [(0, 0), (0, 1)]):
        a._buffer.append((_scene(coords), "ACTION1", "structural"))
    body = ('pts=[b for _,b in state]\n'
            '    if len(pts)<2: return (0.0, False)\n'
            '    a=pts[1]\n'
            '    others=[c for k,c in enumerate(pts) if k!=1]\n'
            '    d=min(abs(a["x"]-c["x"])+abs(a["y"]-c["y"]) for c in others)\n'
            '    return (-float(d), d==0)')
    src = json.dumps({"type": "code", "source": "def reward_function(state):\n    " + body})
    a._llm = _Seq([src])
    ok = a._try_learn_reward(_scene([(0, 0), (0, 6)]))
    assert ok is True
    assert a._reward_fn is not None
