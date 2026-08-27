import numpy as np

from agents.causal.perception import parse, object_at


def test_two_objects_ignoring_background():
    grid = np.zeros((5, 5), dtype=int)
    grid[0, 0] = 3            # objeto A (1 célula)
    grid[2:4, 2:4] = 7        # objeto B (bloco 2x2)
    scene = parse(grid.tolist())
    assert len(scene.objects) == 2
    sizes = sorted(o.size for o in scene.objects)
    assert sizes == [1, 4]
    colors = sorted(o.color for o in scene.objects)
    assert colors == [3, 7]


def test_bbox_and_centroid():
    grid = np.zeros((5, 5), dtype=int)
    grid[1:3, 1:4] = 5        # bloco 2 linhas x 3 colunas
    scene = parse(grid.tolist())
    (o,) = scene.objects
    assert o.bbox == (1, 1, 2, 3)          # (min_row, min_col, max_row, max_col)
    assert o.centroid == (1.5, 2.0)        # (row, col) médio
    assert o.size == 6


def test_stack_uses_last_layer():
    layer0 = np.zeros((3, 3), dtype=int).tolist()
    layer1 = np.zeros((3, 3), dtype=int)
    layer1[0, 0] = 9
    scene = parse([layer0, layer1.tolist()])
    assert len(scene.objects) == 1 and scene.objects[0].color == 9


def test_object_at():
    grid = np.zeros((5, 5), dtype=int)
    grid[1, 4] = 4                                    # linha 1, coluna 4 (assimétrico)
    scene = parse(grid.tolist())
    # x=col, y=row: um swap x/y quebraria estas asserções
    assert object_at(scene, x=4, y=1).color == 4
    assert object_at(scene, x=1, y=4) is None         # coordenada trocada -> vazio
    assert object_at(scene, x=0, y=0) is None         # posição vazia -> None
