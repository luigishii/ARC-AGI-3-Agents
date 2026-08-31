from agents.causal.agent import CausalObjectAgent
from agents.causal.policy import Candidate


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def _c(key, has_object=False):
    return Candidate(None, None, None, key, has_object)


# --- varre: escolhe uma; depois de visitá-la, escolhe OUTRA (menos visitada) ---
def test_cover_decide_sweeps(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_COVER="1")
    cands = [_c("A"), _c("B"), _c("C")]
    first = a._cover_decide(cands)
    assert first in ("A", "B", "C")
    a._cover[first] = 1                      # marca como visitada
    second = a._cover_decide(cands)
    assert second != first                   # próxima é outra (menos visitada)


# --- desempate: has_object vem antes de vazio ---
def test_cover_decide_prefers_object(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_COVER="1")
    out = a._cover_decide([_c("empty", False), _c("obj", True)])
    assert out == "obj"


# --- anti-repetição: evita a última key em empate ---
def test_cover_decide_avoids_last(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_COVER="1")
    a._last_key = "P"
    out = a._cover_decide([_c("P", True), _c("Q", True)])
    assert out == "Q"


# --- contagem no close-loop: _track_cover incrementa a key da última ação ---
def test_track_cover_counts(monkeypatch):
    a = _agent(monkeypatch)
    a._last_key = "ACTION1"
    a._track_cover()
    a._track_cover()
    assert a._cover["ACTION1"] == 2


# --- sem last_key não conta ---
def test_track_cover_no_last(monkeypatch):
    a = _agent(monkeypatch)
    a._last_key = None
    a._track_cover()
    assert a._cover == {}


# --- phase2_stats expõe cover_keys ---
def test_phase2_has_cover_keys(monkeypatch):
    a = _agent(monkeypatch)
    a._cover = {"A": 3, "B": 1}
    assert a.phase2_stats()["cover_keys"] == 2
