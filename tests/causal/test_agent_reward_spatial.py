import numpy as np

from agents.causal.agent import _spatial_context, CausalObjectAgent
from agents.causal.perception import parse, match_objects


def _scene(coords):
    """coords: lista de (row, col) onde plantar um pixel cor 3 (objetos isolados)."""
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


# --- Task 1: _spatial_context ---
def test_spatial_context_has_xy_per_object():
    scene = _scene([(0, 0), (10, 12)])
    txt = _spatial_context(scene.objects)
    assert "x=0" in txt and "y=0" in txt
    assert "x=12" in txt and "y=10" in txt


def test_spatial_context_has_pairwise_distance():
    scene = _scene([(0, 0), (10, 12)])
    txt = _spatial_context(scene.objects)
    assert "DISTANCIAS" in txt
    assert "=22" in txt                            # |0-12|+|0-10| = 22


def test_spatial_context_has_grid_summary():
    scene = _scene([(0, 0), (10, 12)])
    txt = _spatial_context(scene.objects)
    assert "GRID 3x3" in txt


def test_spatial_context_single_object_no_distance_section():
    scene = _scene([(5, 5)])
    txt = _spatial_context(scene.objects)
    assert "x=5" in txt and "y=5" in txt
    assert "DISTANCIAS" not in txt


# --- Task 2: _build_reward_prompt + aceitação ---
def test_reward_prompt_shows_positions_and_fewshot(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (10, 12)])
    p = a._build_reward_prompt(scene)
    assert "x=0" in p and "y=10" in p
    assert "DISTANCIAS" in p
    assert "manhattan" in p.lower() or "x']" in p


def test_spatial_reward_is_accepted(monkeypatch):
    import json
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    for coords in ([(0, 0), (0, 8)], [(0, 0), (0, 4)], [(0, 0), (0, 1)]):
        a._buffer.append((_scene(coords), "ACTION1", "structural"))
    body = ('pts=[b for _,b in state]\n'
            '    if len(pts)<2: return (0.0, False)\n'
            '    a=pts[0]\n'
            '    d=min(abs(a["x"]-b["x"])+abs(a["y"]-b["y"]) for b in pts[1:])\n'
            '    return (-float(d), d==0)')
    src = json.dumps({"type": "code", "source": "def reward_function(state):\n    " + body})
    a._llm = _Seq([src])
    ok = a._try_learn_reward(_scene([(0, 0), (0, 6)]))
    assert ok is True
    assert a._reward_fn is not None


def test_constant_reward_still_rejected(monkeypatch):
    import json
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    for coords in ([(0, 0)], [(0, 2)], [(0, 4)]):
        a._buffer.append((_scene(coords), "ACTION1", "structural"))
    src = json.dumps({"type": "code",
                      "source": "def reward_function(state):\n    _ = len(state)\n    return (0, False)"})
    a._llm = _Seq([src])
    ok = a._try_learn_reward(_scene([(0, 0)]))
    assert ok is False
    assert a._reward_fn is None
    assert a._reward_rejected >= 1
