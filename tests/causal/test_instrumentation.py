import json

from agents.causal.causal_model import Effect
from agents.causal.instrumentation import Instrumentation


def test_logs_accumulate_and_summarize():
    ins = Instrumentation()
    ins.log("ACTION1", None, None, "EXPLORE",
            None, Effect("moved", (0, 1)), {"coverage_keys": 1}, {"why": "novo"})
    ins.log("ACTION1", None, None, "EXPLOIT",
            Effect("none", None), Effect("none", None), {"coverage_keys": 1}, {"why": "x"})
    s = ins.summary()
    assert s["n_actions"] == 2
    assert s["explore_vs_exploit"] == {"EXPLORE": 1, "EXPLOIT": 1}
    assert s["wasted"] == 1


def test_writes_jsonl(tmp_path):
    p = tmp_path / "log.jsonl"
    ins = Instrumentation(str(p))
    ins.log("ACTION2", 3, 4, "EXPLORE", None, Effect("appeared", 5), {}, {})
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["action"] == "ACTION2" and rec["x"] == 3 and rec["actual"] == "appeared"
