from agents.causal.planning import TransitionModel


def test_observe_and_predict_modal():
    m = TransitionModel()
    m.observe("A", "k1", "s1")
    m.observe("A", "k1", "s1")
    m.observe("A", "k1", "s2")     # s1 é modal (2 vs 1)
    assert m.predict_next("A", "k1") == "s1"


def test_predict_unknown_is_none():
    m = TransitionModel()
    assert m.predict_next("A", "k1") is None


def test_known_keys():
    m = TransitionModel()
    m.observe("A", "k1", "s1")
    m.observe("A", "k2", "s2")
    assert set(m.known_keys("A")) == {"k1", "k2"}
    assert m.known_keys("Z") == []


def test_roundtrip_serialization():
    m = TransitionModel()
    m.observe("A", "k1", "s1")
    d = m.to_dict()
    m2 = TransitionModel.from_dict(d)
    assert m2.to_dict() == d
    assert m2.predict_next("A", "k1") == "s1"
