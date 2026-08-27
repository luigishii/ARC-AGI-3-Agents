import json
import os

from analysis.causal_metrics import plot_causal_metrics, summarize_causal_log


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_records():
    return [
        {
            "action": "ACTION1",
            "x": None,
            "y": None,
            "mode": "EXPLORE",
            "predicted": None,
            "actual": "none",
            "model_stats": {"prediction_accuracy": 0.0, "coverage_keys": 1},
            "reasoning": {"key": "k1"},
        },
        {
            "action": "ACTION2",
            "x": 3,
            "y": 4,
            "mode": "EXPLOIT",
            "predicted": "moved",
            "actual": "moved",
            "model_stats": {"prediction_accuracy": 0.5, "coverage_keys": 2},
            "reasoning": {"key": "k2"},
        },
        {
            "action": "ACTION1",
            "x": None,
            "y": None,
            "mode": "EXPLOIT",
            "predicted": "moved",
            "actual": "none",
            "model_stats": {"prediction_accuracy": 0.66, "coverage_keys": 2},
            "reasoning": {"key": "k1"},
        },
    ]


def test_summarize_causal_log(tmp_path):
    p = tmp_path / "causal.jsonl"
    _write_jsonl(p, _make_records())

    summary = summarize_causal_log(str(p))

    assert summary["n_actions"] == 3
    assert summary["wasted"] == 2
    assert summary["explore_vs_exploit"] == {"EXPLORE": 1, "EXPLOIT": 2}
    assert summary["final_prediction_accuracy"] == 0.66
    assert summary["final_coverage_keys"] == 2


def test_plot_causal_metrics_writes_png(tmp_path):
    jsonl_path = tmp_path / "causal.jsonl"
    _write_jsonl(jsonl_path, _make_records())
    out_path = tmp_path / "out" / "causal_metrics.png"

    plot_causal_metrics(str(jsonl_path), str(out_path))

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
