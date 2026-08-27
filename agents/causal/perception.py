# agents/causal/perception.py
from __future__ import annotations

import hashlib
import itertools
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

_id_counter = itertools.count(1)
IOU_THRESHOLD = 0.3


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
    metrics: dict = None       # percepção tripla: grid+grafo(objects)+métricas globais


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


def canonical_color_map(grid: np.ndarray) -> dict:
    """Mapa cor→rank por frequência decrescente (mais comum=0). Torna a física
    aprendida abstrata: gravidade é a mesma pro bloco azul ou amarelo."""
    vals, counts = np.unique(grid, return_counts=True)
    order = sorted(range(len(vals)), key=lambda i: (-counts[i], int(vals[i])))
    return {int(vals[i]): rank for rank, i in enumerate(order)}


def canonical_grid(grid: np.ndarray) -> np.ndarray:
    """Grid recolorido pelos ranks de frequência (invariante à paleta absoluta)."""
    cmap = canonical_color_map(grid)
    out = np.empty_like(grid)
    for orig, rank in cmap.items():
        out[grid == orig] = rank
    return out


def global_metrics(grid: np.ndarray) -> dict:
    """Métricas globais p/ a percepção tripla (grid + grafo + métricas)."""
    vals, counts = np.unique(grid, return_counts=True)
    color_counts = {int(v): int(c) for v, c in zip(vals, counts)}
    bg = int(vals[counts.argmax()])
    rarest = int(vals[counts.argmin()])
    area = int(grid.size - color_counts[bg])          # células não-fundo
    return {
        "color_counts": color_counts,
        "n_colors": len(vals),
        "area": area,
        "bg": bg,
        "rarest_color": rarest,
        "symmetry_h": bool(np.array_equal(grid, np.fliplr(grid))),
        "symmetry_v": bool(np.array_equal(grid, np.flipud(grid))),
    }


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
    return Scene(objects=objects, grid=grid, metrics=global_metrics(grid))


def _iou(a, b) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def match_objects(prev: Scene | None, curr: Scene) -> Scene:
    prev_objs = list(prev.objects) if prev is not None else []
    used = set()
    matched = {}  # indice em curr.objects -> id do prev

    # tier 1: exato cor + shape_hash + centroide mais proximo
    for i, o in enumerate(curr.objects):
        best, best_d = None, None
        for p in prev_objs:
            if p.id in used or p.shape_hash != o.shape_hash or p.color != o.color:
                continue
            d = (p.centroid[0] - o.centroid[0]) ** 2 + (p.centroid[1] - o.centroid[1]) ** 2
            if best_d is None or d < best_d:
                best, best_d = p, d
        if best is not None:
            used.add(best.id)
            matched[i] = best.id

    # tier 2: IoU mesma cor  /  tier 3: IoU qualquer cor
    for require_color in (True, False):
        for i, o in enumerate(curr.objects):
            if i in matched:
                continue
            best, best_iou = None, None
            for p in prev_objs:
                if p.id in used:
                    continue
                if require_color and p.color != o.color:
                    continue
                iou = _iou(p.cells, o.cells)
                if iou >= IOU_THRESHOLD and (best_iou is None or iou > best_iou):
                    best, best_iou = p, iou
            if best is not None:
                used.add(best.id)
                matched[i] = best.id

    new_objs = []
    for i, o in enumerate(curr.objects):
        oid = matched.get(i)
        if oid is None:
            oid = next(_id_counter)
        new_objs.append(replace(o, id=oid))
    return Scene(objects=new_objs, grid=curr.grid, metrics=curr.metrics)


def object_at(scene: Scene, x: int, y: int):
    for o in scene.objects:
        if (y, x) in o.cells:      # x=col, y=row
            return o
    return None
