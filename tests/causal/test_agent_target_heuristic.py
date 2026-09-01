from types import SimpleNamespace as NS

import numpy as np

from agents.causal.agent import _pick_target, CausalObjectAgent
from agents.causal.perception import parse, match_objects


def _obj(color, size, bbox, centroid):
    return NS(color=color, size=size, bbox=bbox, centroid=centroid)


def test_pick_target_rare_compact():
    objs = [
        _obj(9, 15, (0, 0, 2, 4), (1, 2)),        # 0 avatar
        _obj(2, 400, (10, 10, 30, 30), (20, 20)),  # 1 fundo (maior) -> excluido
        _obj(5, 10, (5, 0, 5, 9), (5, 4)),        # 2 barra 1x10 -> excluida
        _obj(4, 4, (8, 8, 9, 9), (8, 8)),         # 3 comum
        _obj(4, 4, (8, 20, 9, 21), (8, 20)),      # 4 comum
        _obj(7, 4, (2, 2, 3, 3), (2, 2)),         # 5 raro compacto
    ]
    assert _pick_target(objs, 0) == 5


def test_pick_target_excludes_bar():
    objs = [
        _obj(9, 15, (0, 0, 2, 4), (1, 2)),        # avatar
        _obj(5, 10, (5, 0, 5, 9), (5, 4)),        # barra raro-mas-alongada -> excluida
        _obj(4, 4, (8, 8, 9, 9), (8, 8)),
        _obj(4, 4, (8, 20, 9, 21), (8, 20)),
        _obj(4, 4, (8, 30, 9, 31), (8, 30)),
    ]
    assert _pick_target(objs, 0) in (2, 3, 4)


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


def _scene(coords):
    g = np.zeros((16, 16), dtype=int)
    for (r, c) in coords:
        g[r, c] = 3
    return match_objects(None, parse(g))


def test_prompt_has_target_hint_and_explicit_fewshot(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (0, 8)])
    objs = list(scene.objects)
    a._move.avatar_counts = {objs[0].id: 5}       # avatar = indice 0
    p = a._build_reward_prompt(scene)
    assert "ALVO PROVAVEL = state[" in p
    assert "t=pts[" in p                           # few-shot explicito avatar->alvo


def test_prompt_fallback_when_no_avatar(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (0, 8)])
    p = a._build_reward_prompt(scene)
    assert "ALVO PROVAVEL" not in p
    assert "a=pts[0]" in p
