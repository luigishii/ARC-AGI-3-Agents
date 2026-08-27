from __future__ import annotations

import numpy as np

HUD_THRESHOLD = 0.7
HUD_MIN_SAMPLES = 5


class HudMask:
    """Detecta o HUD por independência-de-ação em bandas (linha/coluna).

    Um HUD (contador/timer/barra) muda a cada ação, independente de onde se clica.
    Detecção por BANDA (não por célula): uma linha ou coluna que muda em >= HUD_THRESHOLD
    das transições é HUD. Isso captura tanto contadores fixos quanto marcadores que
    "rolam" pela banda (cujo per-célula é baixo, mas a linha muda sempre) — o caso real
    observado no vc33. `mask[r,c]` = linha r é HUD OU coluna c é HUD.
    """

    def __init__(self, shape=(64, 64)):
        self._shape = tuple(shape)
        self.row_count = np.zeros(self._shape[0], dtype=int)
        self.col_count = np.zeros(self._shape[1], dtype=int)
        self.total = 0

    def update(self, prev_grid, curr_grid) -> None:
        a = np.asarray(prev_grid)
        b = np.asarray(curr_grid)
        if a.shape != self._shape:
            # (re)inicializa se a grade tem outro tamanho
            self._shape = a.shape
            self.row_count = np.zeros(self._shape[0], dtype=int)
            self.col_count = np.zeros(self._shape[1], dtype=int)
            self.total = 0
        changed = a != b
        self.row_count += changed.any(axis=1).astype(int)
        self.col_count += changed.any(axis=0).astype(int)
        self.total += 1

    def mask(self) -> np.ndarray:
        if self.total < HUD_MIN_SAMPLES:
            return np.zeros(self._shape, dtype=bool)
        row_hud = (self.row_count / self.total) >= HUD_THRESHOLD
        col_hud = (self.col_count / self.total) >= HUD_THRESHOLD
        return row_hud[:, None] | col_hud[None, :]

    def to_dict(self) -> dict:
        return {
            "shape": list(self._shape),
            "row_count": self.row_count.tolist(),
            "col_count": self.col_count.tolist(),
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HudMask":
        h = cls(tuple(d["shape"]))
        h.row_count = np.array(d["row_count"], dtype=int)
        h.col_count = np.array(d["col_count"], dtype=int)
        h.total = d["total"]
        return h
