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


def parse(frame) -> Scene:
    grid = _to_grid(frame)
    bg = _background_color(grid)
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or grid[r, c] == bg:
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
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] == color:
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
    new_objs = []
    for o in curr.objects:
        best = None
        best_d = None
        for p in prev_objs:
            if p.id in used or p.color != o.color or p.shape_hash != o.shape_hash:
                continue
            d = (p.centroid[0] - o.centroid[0]) ** 2 + (p.centroid[1] - o.centroid[1]) ** 2
            if best_d is None or d < best_d:
                best, best_d = p, d
        if best is not None:
            used.add(best.id)
            new_objs.append(replace(o, id=best.id))
        else:
            new_objs.append(replace(o, id=next(_id_counter)))
    return Scene(objects=new_objs, grid=curr.grid)


def object_at(scene: Scene, x: int, y: int):
    for o in scene.objects:
        if (y, x) in o.cells:      # x=col, y=row
            return o
    return None
