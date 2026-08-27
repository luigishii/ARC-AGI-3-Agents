from agents.causal.policy import GRID_N, cell_center, cell_of


def test_grid_n_is_six():
    assert GRID_N == 6


def test_cell_center_in_bounds_and_distinct():
    seen = set()
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            x, y = cell_center(gx, gy)
            assert 0 <= x <= 63 and 0 <= y <= 63
            assert (x, y) not in seen
            seen.add((x, y))
    assert len(seen) == GRID_N * GRID_N


def test_cell_centers_expected_values():
    # (gx+0.5)*64/6 truncado: {5,16,26,37,48,58}
    xs = sorted({cell_center(gx, 0)[0] for gx in range(GRID_N)})
    assert xs == [5, 16, 26, 37, 48, 58]


def test_cell_of_roundtrips_center():
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            x, y = cell_center(gx, gy)
            assert cell_of(x, y) == (gx, gy)


def test_cell_of_clamps_edges():
    assert cell_of(0, 0) == (0, 0)
    assert cell_of(63, 63) == (GRID_N - 1, GRID_N - 1)
