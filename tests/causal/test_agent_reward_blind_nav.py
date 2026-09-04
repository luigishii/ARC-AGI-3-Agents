"""Modo cego + jogo de teclado: a reward grounded NAO deve congelar no fallback antes do
avatar ser aprendido; e quando o avatar aparece, a reward fallback e trocada pela de
navegacao (-dist avatar->alvo)."""
from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent


class _Frame:
    def __init__(self, frame, available):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = 0
        self.available_actions = available
        self.full_reset = False


def _grid():
    g = [[0] * 64 for _ in range(64)]
    for r in range(10, 13):
        for c in range(10, 13):
            g[r][c] = 9            # avatar
    for r in range(40, 42):
        for c in range(40, 42):
            g[r][c] = 5            # alvo (raro, compacto)
    for r in range(20, 22):
        for c in range(20, 24):
            g[r][c] = 5            # outro objeto cor 5 (mesma cor -> multi-align seria o fallback)
    return [g]


KB = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]


def _agent(monkeypatch, **env):
    env.setdefault("CAUSAL_LLM", "0")
    env.setdefault("CAUSAL_GK", "0")
    env.setdefault("CAUSAL_GROUNDED", "1")
    env.setdefault("CAUSAL_MAX_ACTIONS", "10000")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.game_id = "zzzz"
    a.MAX_ACTIONS = 10000
    a._init_causal_state()
    return a


def _learn_avatar(a, frame):
    """Injeta o avatar aprendido (id do objeto cor 9 na cena DO AGENTE)."""
    aid = next(o.id for o in a._prev_scene.objects if o.color == 9)
    a._move.avatar_counts[aid] = 5
    a._move.vec["ACTION1"] = {(1, 0): 5}
    return aid


def test_grounded_defers_on_keyboard_until_avatar(monkeypatch):
    a = _agent(monkeypatch)
    for _ in range(3):
        a.choose_action([], _Frame(_grid(), KB))
        a.action_counter += 1
    assert a._reward_fn is None                     # esperando o avatar (jogo de teclado)


def test_grounded_deadline_then_fallback(monkeypatch):
    a = _agent(monkeypatch)
    for _ in range(40):
        a.choose_action([], _Frame(_grid(), KB))
        a.action_counter += 1
    assert a._reward_fn is not None                 # deadline: cai no fallback
    assert not a._reward_src.startswith("grounded:-dist")


def test_grounded_click_only_synthesizes_immediately(monkeypatch):
    a = _agent(monkeypatch)
    a.choose_action([], _Frame(_grid(), [GameAction.ACTION6]))
    assert a._reward_fn is not None                 # sem teclado: nao espera avatar


def test_fallback_upgraded_when_avatar_learned(monkeypatch):
    a = _agent(monkeypatch)
    for _ in range(40):
        a.choose_action([], _Frame(_grid(), KB))
        a.action_counter += 1
    assert not a._reward_src.startswith("grounded:-dist")
    _learn_avatar(a, _grid())
    a.choose_action([], _Frame(_grid(), KB))
    # o src passou a carregar o tamanho-ancora: "-dist(cor9#15->cor5#4)"
    assert a._reward_src.startswith("grounded:-dist(cor9#")
    src = a._reward_src
    a.choose_action([], _Frame(_grid(), KB))
    assert a._reward_src == src                     # upgrade so 1x


def _grid_many():
    """>8 objetos antes do avatar na ordem de varredura + avatar COMPOSTO (9 + 12)."""
    g = [[0] * 64 for _ in range(64)]
    for k in range(10):                       # 10 pontinhos cor 8 no topo (indices 0..9)
        g[1][2 + 5 * k] = 8
    for r in range(30, 33):
        for c in range(30, 33):
            g[r][c] = 9                       # avatar parte A (cor 9)
    for r in range(30, 32):
        for c in range(33, 35):
            g[r][c] = 12                      # avatar parte B (cor 12, rara e compacta!)
    for r in range(50, 52):
        for c in range(50, 52):
            g[r][c] = 5                       # alvo real (cor 5, compacto)
    for r in range(10, 12):
        for c in range(40, 44):
            g[r][c] = 5
    return [g]


def test_nav_reward_finds_avatar_beyond_8_and_skips_companion(monkeypatch):
    a = _agent(monkeypatch)
    for _ in range(40):
        a.choose_action([], _Frame(_grid_many(), KB))
        a.action_counter += 1
    aid = next(o.id for o in a._prev_scene.objects if o.color == 9)
    cid = next(o.id for o in a._prev_scene.objects if o.color == 12)
    a._move.avatar_counts[aid] = 5
    a._move.vec["ACTION1"] = {(1, 0): 5}
    a._move.companions[aid] = {cid}
    a.choose_action([], _Frame(_grid_many(), KB))
    # avatar cor 9 (bloco 3x3 = 9px), alvo cor 5 (2x2 = 4px); o tamanho e a ancora
    # que desempata quando varios objetos compartilham a cor.
    assert a._reward_src == "grounded:-dist(cor9#9->cor5#4)", a._reward_src
