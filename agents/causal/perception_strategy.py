# agents/causal/perception_strategy.py
# §2 — PerceptionStrategy com fallback pra grid bruto (aviso do Tycho).
# A percepção objeto-cêntrica é um viés forte; muitos jogos são melhor
# modelados como UI/autômato/grid. Se a contagem/tracking de objetos oscila
# entre frames, degradamos suavemente pra assinatura de grid bruto — evita
# travar em jogos sem objetos estáveis.
from __future__ import annotations

from collections import deque

import numpy as np

from .novelty import state_signature
from .policy import GRID_N


class PerceptionStrategy:
    """Detecta instabilidade do grafo de objetos numa janela deslizante."""

    def __init__(self, window: int = 4, tol: int = 1):
        self._counts = deque(maxlen=window)
        self.tol = tol

    def observe(self, n_objects: int) -> None:
        self._counts.append(int(n_objects))

    def stable(self) -> bool:
        if len(self._counts) < 2:
            return True
        return (max(self._counts) - min(self._counts)) <= self.tol

    def mode(self) -> str:
        return "objects" if self.stable() else "grid"


def grid_signature(grid) -> str:
    """Assinatura de fallback: grid bruto reduzido a uma malha GRID_N×GRID_N pela
    cor modal de cada célula (invariante a ruído fino, sem depender de objetos)."""
    if grid is None:
        return ""
    arr = np.asarray(grid)
    rows, cols = arr.shape
    parts = []
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            r0, r1 = gy * rows // GRID_N, (gy + 1) * rows // GRID_N
            c0, c1 = gx * cols // GRID_N, (gx + 1) * cols // GRID_N
            block = arr[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)]
            vals, counts = np.unique(block, return_counts=True)
            parts.append(str(int(vals[counts.argmax()])))
    return ",".join(parts)


def signature_for(scene, strategy: PerceptionStrategy) -> str:
    """Assinatura de estado escolhida pela estratégia: objeto-cêntrica quando o
    grafo é estável, grid bruto quando oscila (fallback do Tycho)."""
    if strategy.mode() == "grid" and getattr(scene, "grid", None) is not None:
        return grid_signature(scene.grid)
    return state_signature(scene)
