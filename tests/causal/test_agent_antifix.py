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


def _keymap(cands):
    return {c.key: c for c in cands}


# --- quebra na K-ésima repetição, sobrepondo por candidata != key fixada ---
def test_antifix_breaks_after_k(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_FIX="1", CAUSAL_FIX_K="3")
    a._last_key = "A"
    cands = [_c("A"), _c("B")]
    km = _keymap(cands)
    out1 = a._antifix(_c("A"), cands, km)
    out2 = a._antifix(_c("A"), cands, km)
    out3 = a._antifix(_c("A"), cands, km)
    assert out1.key == "A" and out2.key == "A"
    assert out3.key == "B"                     # 3ª repetição -> sobrepõe
    assert a._fix_breaks == 1


# --- abaixo de K não quebra ---
def test_antifix_below_k(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_FIX="1", CAUSAL_FIX_K="3")
    a._last_key = "A"
    cands = [_c("A"), _c("B")]
    km = _keymap(cands)
    a._antifix(_c("A"), cands, km)
    out = a._antifix(_c("A"), cands, km)
    assert out.key == "A"
    assert a._fix_breaks == 0


# --- key diferente da anterior zera o run (nunca quebra alternando) ---
def test_antifix_diff_key_resets(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_FIX="1", CAUSAL_FIX_K="3")
    a._last_key = "A"
    cands = [_c("A"), _c("B")]
    km = _keymap(cands)
    for _ in range(5):
        out = a._antifix(_c("B"), cands, km)   # B != _last_key "A" -> run sempre 0
    assert out.key == "B"
    assert a._fix_breaks == 0


# --- sem alternativa (só a key fixada) não força ---
def test_antifix_no_alt(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_FIX="1", CAUSAL_FIX_K="3")
    a._last_key = "A"
    cands = [_c("A")]
    km = _keymap(cands)
    for _ in range(5):
        out = a._antifix(_c("A"), cands, km)
    assert out.key == "A"
    assert a._fix_breaks == 0


# --- off por default: nunca sobrepõe ---
def test_antifix_off_default(monkeypatch):
    a = _agent(monkeypatch)                     # CAUSAL_FIX não setado
    a._last_key = "A"
    cands = [_c("A"), _c("B")]
    km = _keymap(cands)
    for _ in range(5):
        out = a._antifix(_c("A"), cands, km)
    assert out.key == "A"
    assert a._fix_breaks == 0


# --- phase2_stats expõe fix_breaks ---
def test_phase2_has_fix_breaks(monkeypatch):
    a = _agent(monkeypatch)
    a._fix_breaks = 5
    assert a.phase2_stats()["fix_breaks"] == 5
