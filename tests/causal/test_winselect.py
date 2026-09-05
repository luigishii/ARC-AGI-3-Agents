import math
from agents.causal.winselect import explain_score, select_win_reward


def _states(n):
    # estados sinteticos: 1 objeto cuja coordenada x cresce com o tempo
    return [[("h", {"x": i, "y": 0, "color": 1, "shape": "h", "size": 4})] for i in range(n)]


def test_explain_score_argmax_and_rho():
    is_top, rho = explain_score([1.0, 2.0, 3.0, 4.0], win_idx=3)
    assert is_top and rho > 0.9


def test_explain_score_not_top():
    is_top, rho = explain_score([5.0, 2.0, 3.0, 4.0], win_idx=3)
    assert not is_top


def test_explain_score_constant_is_zero_rho():
    is_top, rho = explain_score([1.0, 1.0, 1.0], win_idx=2)
    assert is_top and rho == 0.0


def test_select_picks_increasing_argmax_candidate():
    st = _states(5); win = st[-1]; level = st[:-1]
    up = ("up", lambda s: (float(s[0][1]["x"]), False))          # cresce, win e argmax
    down = ("down", lambda s: (-float(s[0][1]["x"]), False))     # decresce
    const = ("const", lambda s: (1.0, False))                    # sem gradiente
    got = select_win_reward([down, const, up], level, win)
    assert got is not None and got[0] == "up" and got[2] > 0.9


def test_select_skips_exception_and_nonfinite():
    st = _states(5); win = st[-1]; level = st[:-1]
    boom = ("boom", lambda s: 1 / 0)
    nan = ("nan", lambda s: (math.nan, False))
    up = ("up", lambda s: (float(s[0][1]["x"]), False))
    assert select_win_reward([boom, nan, up], level, win)[0] == "up"


def test_select_none_when_no_candidate_explains():
    st = _states(5); win = st[-1]; level = st[:-1]
    down = ("down", lambda s: (-float(s[0][1]["x"]), False))
    assert select_win_reward([down], level, win) is None


def test_select_none_with_too_few_states():
    st = _states(3); win = st[-1]; level = st[:-1]       # 2 estados de nivel < 3
    up = ("up", lambda s: (float(s[0][1]["x"]), False))
    assert select_win_reward([up], level, win) is None
