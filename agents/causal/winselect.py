"""Seleção da reward que EXPLICA a 1ª vitória de um jogo.

Dada a trajetória rotulada de um nível (estados decisão→decisão + estado vencedor),
uma reward candidata "explica" a vitória se (1) o estado vencedor é o de maior valor
(empate permitido) e (2) o valor cresce com o tempo (Spearman > 0). Entre as válidas,
vence a de maior Spearman. Puro numpy/stdlib; exception-safe por candidato.

Medido nas gravações (05/set): no L0 do vc33 e do tn36 a `grounded_multi_reward_fn`
tem o frame vencedor como argmax (rank 1/39 e 1/89) com Spearman +0.83 / +0.93 —
ela explicava a vitória, mas o level-up trocava a reward pelo template(win-grid).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

MIN_LEVEL_STATES = 3


def _spearman(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    v = np.asarray(values, dtype=float)
    t = np.arange(len(v), dtype=float)
    if np.std(v) == 0.0:
        return 0.0
    rv = np.argsort(np.argsort(v)).astype(float)
    rt = np.argsort(np.argsort(t)).astype(float)
    rho = float(np.corrcoef(rt, rv)[0, 1])
    return 0.0 if math.isnan(rho) else rho


def explain_score(values: list[float], win_idx: int) -> tuple[bool, float]:
    """(is_top, rho): o valor em win_idx é >= todos os outros? e Spearman(tempo, valor)."""
    if not values:
        return (False, 0.0)
    win = values[win_idx]
    is_top = all(win >= v for v in values)
    return (is_top, _spearman(values))


def _value(fn: Callable, state) -> float | None:
    try:
        r = fn(state)
    except Exception:
        return None
    v = r[0] if isinstance(r, (tuple, list)) else r
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def select_win_reward(candidates: list[tuple[str, Callable]], level_states: list,
                      win_state) -> tuple[str, Callable, float] | None:
    """Melhor candidato que explica a vitória, ou None (sem evidência/ninguém explica)."""
    if len(level_states) < MIN_LEVEL_STATES:
        return None
    traj = list(level_states) + [win_state]
    best = None
    for name, fn in candidates:
        vals = []
        for st in traj:
            v = _value(fn, st)
            if v is None:
                vals = None
                break
            vals.append(v)
        if not vals or len({round(v, 6) for v in vals}) < 2:
            continue
        is_top, rho = explain_score(vals, len(vals) - 1)
        if not is_top or rho <= 0.0:
            continue
        if best is None or rho > best[2]:
            best = (name, fn, rho)
    return best
