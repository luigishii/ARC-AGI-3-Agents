import numpy as np

from agents.causal.llm import build_direct_prompt, grid_to_ascii
from agents.causal.perception import parse


def test_grid_to_ascii_hex_per_cell():
    # cada celula vira 1 char hex (cor 0-15 -> 0-f); linhas separadas por \n
    assert grid_to_ascii([[0, 15], [3, 10]]) == "0f\n3a"


def test_grid_to_ascii_handles_none():
    assert grid_to_ascii(None) == ""


def test_build_direct_prompt_includes_ascii_grid():
    grid = np.zeros((8, 8), dtype=int)
    grid[2, 3] = 5
    scene = parse(grid.tolist())
    p = build_direct_prompt(scene, {"available": ["ACTION6"]})
    assert "GRID" in p                       # cabecalho do grid
    assert grid_to_ascii(scene.grid) in p    # o desenho ASCII do grid entra no prompt
