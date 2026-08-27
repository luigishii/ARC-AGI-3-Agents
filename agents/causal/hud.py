from __future__ import annotations

import numpy as np

HUD_THRESHOLD = 0.7
HUD_MIN_SAMPLES = 5


class HudMask:
    def __init__(self, shape=(64, 64)):
        self._shape = tuple(shape)
        self.change_count = np.zeros(self._shape, dtype=int)
        self.total = 0

    def update(self, prev_grid, curr_grid) -> None:
        a = np.asarray(prev_grid)
        b = np.asarray(curr_grid)
        if a.shape != self.change_count.shape:
            # (re)inicializa se a grade tem outro tamanho
            self._shape = a.shape
            self.change_count = np.zeros(self._shape, dtype=int)
            self.total = 0
        self.change_count += (a != b).astype(int)
        self.total += 1

    def mask(self) -> np.ndarray:
        if self.total < HUD_MIN_SAMPLES:
            return np.zeros(self._shape, dtype=bool)
        return (self.change_count / self.total) >= HUD_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "shape": list(self._shape),
            "change_count": self.change_count.tolist(),
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HudMask":
        h = cls(tuple(d["shape"]))
        h.change_count = np.array(d["change_count"], dtype=int)
        h.total = d["total"]
        return h
