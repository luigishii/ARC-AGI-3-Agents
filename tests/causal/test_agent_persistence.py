import os
from agents.causal.agent import CausalObjectAgent
from agents.causal.transfer import (
    TransferPrior, save_prior, reset_shared_prior, load_prior,
)


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._cleanup = False               # neutraliza o cleanup do base (sem API)
    a._init_causal_state()
    return a


def _seed(tmp_path):
    path = str(tmp_path / "prior.json")
    p = TransferPrior()
    p._counts = {"click_on_object": [9, 10]}
    save_prior(p, path)
    return path


def test_init_loads_prior_from_env(tmp_path, monkeypatch):
    reset_shared_prior()
    path = _seed(tmp_path)
    monkeypatch.setenv("CAUSAL_PRIOR", path)
    a = _agent()
    assert a._prior.productivity("click_on_object") == 0.9


def test_cleanup_saves_when_flag_set(tmp_path, monkeypatch):
    reset_shared_prior()
    path = str(tmp_path / "out.json")
    monkeypatch.setenv("CAUSAL_PRIOR", path)
    monkeypatch.setenv("CAUSAL_PRIOR_SAVE", "1")
    a = _agent()
    a._prior.observe("simple", "moved")
    a.cleanup()
    assert load_prior(path) is not None
    assert load_prior(path)._counts.get("simple") == [1, 1]


def test_cleanup_does_not_save_without_flag(tmp_path, monkeypatch):
    reset_shared_prior()
    path = str(tmp_path / "out.json")
    monkeypatch.setenv("CAUSAL_PRIOR", path)
    monkeypatch.delenv("CAUSAL_PRIOR_SAVE", raising=False)
    a = _agent()
    a._prior.observe("simple", "moved")
    a.cleanup()
    assert not os.path.exists(path)
