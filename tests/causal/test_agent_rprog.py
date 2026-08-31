import numpy as np

from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def _scene(n):
    """Cena com n objetos (n pixels NÃO-adjacentes → n componentes conexos separados)."""
    g = np.zeros((8, 8), dtype=int)
    for i in range(n):
        g[0, i * 2] = 3               # colunas 0,2,4,... → não se conectam
    return match_objects(None, parse(g))


# --- tracker acumula Δ>0 quando o reward sobe (menos objetos) ---
def test_track_rprog_positive_delta(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_fn = lambda state: (100.0 - (len(state) - 1) * 10.0, False)
    a._prev_scene = _scene(3)          # antes: 3 objetos → value 80
    a._last_key = "ACTION6"
    a._track_rprog(_scene(1))          # depois: 1 objeto → value 100 → Δ=+20
    assert a._rprog["ACTION6"][1] == 1
    assert a._rprog["ACTION6"][0] == 20.0


# --- tracker acumula Δ<0 quando o reward cai (mais objetos) ---
def test_track_rprog_negative_delta(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_fn = lambda state: (100.0 - (len(state) - 1) * 10.0, False)
    a._prev_scene = _scene(1)
    a._last_key = "ACTION1"
    a._track_rprog(_scene(3))          # 100 → 80 → Δ=-20
    assert a._rprog["ACTION1"][0] == -20.0


# --- sem reward_fn: não rastreia ---
def test_track_rprog_no_reward(monkeypatch):
    a = _agent(monkeypatch)
    assert a._reward_fn is None
    a._prev_scene = _scene(2)
    a._last_key = "ACTION1"
    a._track_rprog(_scene(1))
    assert a._rprog == {}


# --- valor não-finito é descartado (não polui a média) ---
def test_track_rprog_discards_non_finite(monkeypatch):
    a = _agent(monkeypatch)
    def boom(state):
        raise ValueError("boom")       # value_fn_from_reward → -inf
    a._reward_fn = boom
    a._prev_scene = _scene(2)
    a._last_key = "ACTION1"
    a._track_rprog(_scene(1))
    assert a._rprog == {}


# --- phase2_stats expõe rprog_actions e rprog_fires ---
def test_phase2_stats_rprog_keys(monkeypatch):
    a = _agent(monkeypatch)
    a._rprog = {"ACTION6": [30.0, 3], "ACTION1": [-5.0, 2]}
    a._rprog_fires = 7
    s = a.phase2_stats()
    assert s["rprog_actions"] == 1     # só ACTION6 tem média > 0
    assert s["rprog_fires"] == 7
