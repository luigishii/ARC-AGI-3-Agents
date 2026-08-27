# agents/causal/perception.py
from __future__ import annotations

import hashlib
import itertools
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

_id_counter = itertools.count(1)


@dataclass(frozen=True)
class Object:
    color: int
    cells: frozenset          # frozenset[tuple[int,int]] em (row,col)
    bbox: tuple               # (min_row, min_col, max_row, max_col)
    centroid: tuple           # (row, col) float
    size: int
    shape_hash: str
    id: int | None = None


@dataclass
class Scene:
    objects: list
    grid: np.ndarray = field(repr=False)


def _to_grid(frame) -> np.ndarray:
    arr = np.array(frame)
    while arr.ndim > 2:        # pilha de camadas -> última
        arr = arr[-1]
    return arr.astype(int)


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _shape_hash(cells: frozenset) -> str:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return hashlib.md5(str(norm).encode()).hexdigest()[:8]


def to_grid(frame) -> np.ndarray:
    return _to_grid(frame)


def parse(frame, hud_mask=None) -> Scene:
    grid = _to_grid(frame)
    bg = _background_color(grid)
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            masked = hud_mask is not None and hud_mask[r, c]
            if seen[r, c] or grid[r, c] == bg or masked:
                continue
            color = int(grid[r, c])
            cells = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    nmasked = hud_mask is not None and hud_mask[nr, nc]
                    if not seen[nr, nc] and grid[nr, nc] == color and not nmasked:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            cset = frozenset(cells)
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
            objects.append(
                Object(color, cset, bbox, centroid, len(cells), _shape_hash(cset))
            )
    return Scene(objects=objects, grid=grid)


def match_objects(prev: Scene | None, curr: Scene) -> Scene:
    prev_objs = list(prev.objects) if prev is not None else []
    used = set()
    matched = {}  # indice em curr.objects -> id do prev

    def _best(o, require_color):
        best, best_d = None, None
        for p in prev_objs:
            if p.id in used or p.shape_hash != o.shape_hash:
                continue
            if require_color and p.color != o.color:
                continue
            d = (p.centroid[0] - o.centroid[0]) ** 2 + (p.centroid[1] - o.centroid[1]) ** 2
            if best_d is None or d < best_d:
                best, best_d = p, d
        return best, best_d

    # passe 1: match estrito por cor + shape + centroide mais proximo
    for i, o in enumerate(curr.objects):
        best, _ = _best(o, True)
        if best is not None:
            used.add(best.id)
            matched[i] = best.id
    # passe 2: fallback ignorando cor (recolor), so quando muito proximo
    for i, o in enumerate(curr.objects):
        if i in matched:
            continue
        best, best_d = _best(o, False)
        if best is not None and best_d is not None and best_d <= 2.0:
            used.add(best.id)
            matched[i] = best.id

    new_objs = []
    for i, o in enumerate(curr.objects):
        oid = matched.get(i)
        if oid is None:
            oid = next(_id_counter)
        new_objs.append(replace(o, id=oid))
    return Scene(objects=new_objs, grid=curr.grid)


def object_at(scene: Scene, x: int, y: int):
    for o in scene.objects:
        if (y, x) in o.cells:      # x=col, y=row
            return o
    return None
