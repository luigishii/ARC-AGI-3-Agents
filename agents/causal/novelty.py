# agents/causal/novelty.py
from __future__ import annotations

import math

from .policy import cell_of

OPTIMISTIC_YIELD = 1.0


def state_signature(scene) -> str:
    parts = []
    for o in scene.objects:
        gx, gy = cell_of(int(round(o.centroid[1])), int(round(o.centroid[0])))
        parts.append((o.color, gx, gy))
    return ";".join(f"{c},{gx},{gy}" for (c, gx, gy) in sorted(parts))
