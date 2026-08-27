from __future__ import annotations

import json
from collections import Counter


def _kind(effect):
    return None if effect is None else effect.kind


class Instrumentation:
    def __init__(self, path: str | None = None):
        self.path = path
        self.records = []

    def log(self, action_name, x, y, mode, predicted, actual, model_stats, reasoning):
        rec = {
            "action": action_name,
            "x": x,
            "y": y,
            "mode": mode,
            "predicted": _kind(predicted),
            "actual": _kind(actual),
            "model_stats": model_stats,
            "reasoning": reasoning,
        }
        self.records.append(rec)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def summary(self) -> dict:
        modes = Counter(r["mode"] for r in self.records)
        wasted = sum(1 for r in self.records if r["actual"] == "none")
        return {
            "n_actions": len(self.records),
            "explore_vs_exploit": dict(modes),
            "wasted": wasted,
            "last_model_stats": self.records[-1]["model_stats"] if self.records else {},
        }
