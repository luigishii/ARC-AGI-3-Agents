import json
from agents.causal.causal_model import CausalModel, Effect


def test_observe_and_predict_modal_effect():
    m = CausalModel()
    m._bump("ACTION1", Effect("moved", (0, 1)))
    m._bump("ACTION1", Effect("moved", (0, 1)))
    m._bump("ACTION1", Effect("none", None))
    eff, conf = m.predict("ACTION1")
    assert eff.kind == "moved"
    assert abs(conf - 2 / 3) < 1e-9


def test_predict_unknown_key():
    assert CausalModel().predict("ACTION9") == (None, 0.0)


def test_progress_flag():
    m = CausalModel()
    m._bump("ACTION2", Effect("structural", ("disappeared",)), level_up=True)
    assert m.is_progress("ACTION2") is True
    assert m.is_progress("ACTION1") is False


def test_prediction_accuracy_tracking():
    m = CausalModel()
    m.record_prediction(Effect("moved", (0, 1)), Effect("moved", (0, 1)))  # acerto
    m.record_prediction(Effect("none", None), Effect("moved", (0, 1)))     # erro
    assert abs(m.stats()["prediction_accuracy"] - 0.5) < 1e-9


def test_serialization_roundtrip():
    m = CausalModel()
    m._bump("ACTION1", Effect("moved", (0, 1)), level_up=True)
    d = json.loads(json.dumps(m.to_dict()))
    m2 = CausalModel.from_dict(d)
    assert m2.predict("ACTION1")[0].kind == "moved"
    assert m2.is_progress("ACTION1") is True
