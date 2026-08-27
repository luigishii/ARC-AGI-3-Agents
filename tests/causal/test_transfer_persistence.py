import os
import json
from agents.causal.transfer import (
    TransferPrior, save_prior, load_prior, DEFAULT_PRIOR_PATH,
)


def _prior(counts):
    p = TransferPrior()
    p._counts = {k: list(v) for k, v in counts.items()}
    return p


def test_default_prior_path():
    assert DEFAULT_PRIOR_PATH == "agents/causal/prior.json"


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "prior.json")
    p = _prior({"click_on_object": [8, 10], "simple": [2, 2]})
    save_prior(p, path)
    assert os.path.exists(path)
    p2 = load_prior(path)
    assert p2.to_dict() == p.to_dict()
    assert p2.productivity("click_on_object") == 0.8


def test_save_is_atomic_no_tmp_left(tmp_path):
    path = str(tmp_path / "prior.json")
    save_prior(_prior({"simple": [1, 1]}), path)
    assert not os.path.exists(path + ".tmp")
    with open(path) as f:
        json.load(f)                      # JSON válido


def test_save_creates_missing_dir(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "prior.json")
    save_prior(_prior({"simple": [1, 1]}), path)
    assert os.path.exists(path)


def test_load_missing_returns_none(tmp_path):
    assert load_prior(str(tmp_path / "nope.json")) is None
    assert load_prior("") is None


def test_merge_accumulates():
    a = _prior({"simple": [1, 2], "click_empty": [0, 3]})
    b = _prior({"simple": [3, 4], "click_on_object": [5, 5]})
    a.merge(b)
    d = a.to_dict()["counts"]
    assert d["simple"] == [4, 6]
    assert d["click_empty"] == [0, 3]
    assert d["click_on_object"] == [5, 5]
