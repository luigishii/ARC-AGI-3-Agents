import threading
from arcengine import GameAction
from agents.causal.policy import Candidate
from agents.causal.transfer import (
    abstract_feature, TransferPrior, shared_prior, reset_shared_prior,
    NEUTRAL_PRODUCTIVITY,
)


def _simple():
    return Candidate(GameAction.ACTION1, None, None, "ACTION1", False)


def _click(has_object):
    return Candidate(GameAction.ACTION6, 5, 5, "ACTION6@cell=0,0", has_object)


def test_abstract_feature_buckets():
    assert abstract_feature(_simple()) == "simple"
    assert abstract_feature(_click(True)) == "click_on_object"
    assert abstract_feature(_click(False)) == "click_empty"


def test_productivity_neutral_without_data():
    p = TransferPrior()
    assert p.productivity("click_on_object") == NEUTRAL_PRODUCTIVITY


def test_productivity_reflects_counts():
    p = TransferPrior()
    p.observe("click_on_object", "moved")     # produtivo
    p.observe("click_on_object", "none")      # não
    p.observe("click_on_object", None)        # não
    assert p.productivity("click_on_object") == 1 / 3


def test_observe_is_threadsafe():
    p = TransferPrior()

    def worker():
        for _ in range(1000):
            p.observe("simple", "moved")

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert p._counts["simple"] == [4000, 4000]


def test_roundtrip_serialization():
    p = TransferPrior()
    p.observe("simple", "moved")
    p.observe("simple", "none")
    d = p.to_dict()
    p2 = TransferPrior.from_dict(d)
    assert p2.to_dict() == d
    assert p2.productivity("simple") == p.productivity("simple")


def test_singleton_identity_and_reset():
    reset_shared_prior()
    a = shared_prior()
    b = shared_prior()
    assert a is b
    reset_shared_prior()
    assert shared_prior() is not a
