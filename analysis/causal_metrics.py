"""Métricas offline do CausalObjectAgent a partir do log JSONL do Instrumentation.

Módulo standalone (não depende de `analysis/replay.py`, que não existe neste
worktree). Lê o JSONL produzido por `agents.causal.instrumentation.Instrumentation`
(um registro por ação, campos: action, x, y, mode, predicted, actual,
model_stats {prediction_accuracy, coverage_keys, ...}, reasoning) e produz:

- um resumo simples (`summarize_causal_log`), sem dependências além de stdlib;
- um PNG com as curvas de `prediction_accuracy` e `coverage_keys` por passo
  (`plot_causal_metrics`), via matplotlib com backend headless.

Uso:
    uv run python analysis/causal_metrics.py --causal-log analysis/out/causal.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _read_records(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize_causal_log(jsonl_path: str) -> dict:
    """Resumo simples do log causal, sem dependência de matplotlib.

    Retorna: n_actions, final_prediction_accuracy, final_coverage_keys,
    wasted (linhas com actual == "none"), explore_vs_exploit (contagem por mode).
    """
    records = _read_records(jsonl_path)

    n_actions = len(records)
    wasted = sum(1 for r in records if r.get("actual") == "none")
    explore_vs_exploit = dict(Counter(r.get("mode") for r in records))

    last_stats = records[-1].get("model_stats", {}) if records else {}
    final_prediction_accuracy = last_stats.get("prediction_accuracy", 0.0)
    final_coverage_keys = last_stats.get("coverage_keys", 0)

    return {
        "n_actions": n_actions,
        "final_prediction_accuracy": final_prediction_accuracy,
        "final_coverage_keys": final_coverage_keys,
        "wasted": wasted,
        "explore_vs_exploit": explore_vs_exploit,
    }


def plot_causal_metrics(jsonl_path: str, out_path: str) -> None:
    """Plota prediction_accuracy e coverage_keys por passo, salvando PNG em out_path."""
    records = _read_records(jsonl_path)

    acc, cov = [], []
    for rec in records:
        ms = rec.get("model_stats", {})
        acc.append(ms.get("prediction_accuracy", 0.0))
        cov.append(ms.get("coverage_keys", 0))

    fig, ax1 = plt.subplots()
    ax1.plot(acc, label="prediction_accuracy")
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("step")

    ax2 = ax1.twinx()
    ax2.plot(cov, "--", color="tab:orange", label="coverage_keys")
    ax2.set_ylabel("coverage (keys)")

    fig.legend(loc="upper left")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Métricas offline do CausalObjectAgent a partir do log JSONL."
    )
    parser.add_argument("--causal-log", required=True, help="Caminho do JSONL do Instrumentation.")
    parser.add_argument(
        "--out", default="analysis/out/causal_metrics.png", help="Caminho do PNG de saída."
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_causal_metrics(args.causal_log, args.out)
    summary = summarize_causal_log(args.causal_log)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
