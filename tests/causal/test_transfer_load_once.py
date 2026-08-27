from agents.causal.transfer import (
    TransferPrior, save_prior, shared_prior, reset_shared_prior, load_shared_once,
)


def _seed_file(tmp_path):
    path = str(tmp_path / "prior.json")
    p = TransferPrior()
    p._counts = {"click_on_object": [9, 10]}   # produtividade 0.9
    save_prior(p, path)
    return path


def test_load_shared_once_merges_into_singleton(tmp_path):
    reset_shared_prior()
    path = _seed_file(tmp_path)
    load_shared_once(path)
    assert shared_prior().productivity("click_on_object") == 0.9


def test_load_shared_once_runs_only_once(tmp_path):
    reset_shared_prior()
    path = _seed_file(tmp_path)
    load_shared_once(path)
    load_shared_once(path)                       # 2ª vez: no-op (não duplica)
    assert shared_prior()._counts["click_on_object"] == [9, 10]


def test_reset_allows_reload(tmp_path):
    reset_shared_prior()
    path = _seed_file(tmp_path)
    load_shared_once(path)
    reset_shared_prior()                         # zera singleton E _loaded
    load_shared_once(path)
    assert shared_prior()._counts["click_on_object"] == [9, 10]


def test_load_shared_once_missing_file_is_noop(tmp_path):
    reset_shared_prior()
    load_shared_once(str(tmp_path / "nope.json"))
    assert shared_prior()._counts == {}
