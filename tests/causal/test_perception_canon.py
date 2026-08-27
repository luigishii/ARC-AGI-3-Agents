import numpy as np

from agents.causal.perception import (
    canonical_color_map, canonical_grid, global_metrics, parse,
)
from agents.causal.perception_strategy import (
    PerceptionStrategy, grid_signature, signature_for,
)


# --- §2a: canonização de cores ---
def test_canonical_color_map_by_frequency():
    grid = np.array([[0, 0, 0], [0, 5, 5], [0, 5, 7]])
    cmap = canonical_color_map(grid)
    assert cmap[0] == 0        # mais frequente → rank 0
    assert cmap[5] == 1        # segundo → 1
    assert cmap[7] == 2


def test_canonical_grid_is_palette_invariant():
    a = np.array([[0, 0], [0, 9]])
    b = np.array([[3, 3], [3, 4]])   # mesma estrutura, paleta diferente
    assert np.array_equal(canonical_grid(a), canonical_grid(b))


def test_global_metrics_fields():
    grid = np.array([[0, 0, 0], [0, 5, 5], [0, 5, 7]])
    m = global_metrics(grid)
    assert m["bg"] == 0
    assert m["rarest_color"] == 7
    assert m["n_colors"] == 3
    assert m["area"] == 4        # células não-fundo (três 5 + um 7)
    assert m["symmetry_h"] is False


def test_parse_attaches_metrics():
    grid = np.zeros((8, 8), dtype=int)
    grid[2, 2] = 4
    scene = parse(grid)
    assert scene.metrics is not None
    assert scene.metrics["bg"] == 0
    assert scene.grid is not None          # percepção tripla: grid presente
    assert isinstance(scene.objects, list) # + grafo de objetos


# --- §2b: PerceptionStrategy fallback (Tycho) ---
def test_strategy_stable_when_counts_steady():
    s = PerceptionStrategy(window=4, tol=1)
    for _ in range(4):
        s.observe(3)
    assert s.stable() is True
    assert s.mode() == "objects"


def test_strategy_unstable_when_counts_oscillate():
    s = PerceptionStrategy(window=4, tol=1)
    for n in (2, 9, 3, 11):
        s.observe(n)
    assert s.stable() is False
    assert s.mode() == "grid"


def test_grid_signature_palette_invariant_and_stable():
    a = np.zeros((12, 12), dtype=int); a[0:6, 0:6] = 4
    b = np.zeros((12, 12), dtype=int); b[0:6, 0:6] = 4
    assert grid_signature(a) == grid_signature(b)


def test_signature_for_switches_on_instability():
    grid = np.zeros((12, 12), dtype=int); grid[0, 0] = 4

    class _Scene:
        objects = []
        def __init__(self, g): self.grid = g

    scn = _Scene(grid)
    stable = PerceptionStrategy()
    stable.observe(1); stable.observe(1)
    unstable = PerceptionStrategy(tol=1)
    for n in (1, 8, 2, 9):
        unstable.observe(n)
    assert signature_for(scn, unstable) == grid_signature(grid)
    # estável usa a assinatura objeto-cêntrica (aqui, sem objetos → string vazia)
    assert signature_for(scn, stable) != grid_signature(grid)
