"""Level-up com CAUSAL_WINREWARD: a reward do proximo nivel e a candidata que EXPLICA a
vitoria (aqui multi-align: 2 objetos da mesma cor convergem ate encostar), nao o
template(win-grid). Sem o flag, comportamento antigo (template)."""
from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent

CLICK = [GameAction.ACTION6]


class _Frame:
    def __init__(self, frame, levels=0):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = levels
        self.available_actions = CLICK
        self.full_reset = False


def _layer(gap):
    """Fundo 0; bloco A cor 5 fixo em x=10..12; bloco B cor 5 a `gap` colunas a direita."""
    g = [[0] * 64 for _ in range(64)]
    for r in range(30, 33):
        for c in range(10, 13):
            g[r][c] = 5
        for c in range(13 + gap, 16 + gap):
            g[r][c] = 5
    return g


def _agent(monkeypatch, **env):
    env.setdefault("CAUSAL_LLM", "0")
    env.setdefault("CAUSAL_GK", "0")
    env.setdefault("CAUSAL_GROUNDED", "1")
    env.setdefault("CAUSAL_RPROG", "1")
    env.setdefault("CAUSAL_MAX_ACTIONS", "10000")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.game_id = "zzzz"
    a.MAX_ACTIONS = 10000
    a._init_causal_state()
    return a


def _play_level(a):
    """Blocos convergem 20→...→2 (frames do nivel), depois o frame de VITORIA
    (pilha [camada vencedora gap=0, init do proximo nivel gap=20])."""
    for gap in (20, 16, 12, 8, 4, 2):
        a.choose_action([], _Frame([_layer(gap)]))
        a.action_counter += 1
    a.choose_action([], _Frame([_layer(0), _layer(20)], levels=1))
    a.action_counter += 1


def test_levelup_selects_reward_that_explains_win(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_WINREWARD="1")
    _play_level(a)
    assert a._win_reward is not None and a._win_reward[0] == "multi-align"
    assert a._win_reward[2] > 0.5
    a.choose_action([], _Frame([_layer(20)], levels=1))       # 1o passo do L1
    assert a._reward_src.startswith("win:multi-align")
    st = a.phase2_stats()
    assert st["win_reward"] == "multi-align" and st["win_rho"] > 0.5


def test_levelup_without_flag_keeps_template(monkeypatch):
    a = _agent(monkeypatch)
    _play_level(a)
    assert a._win_reward is None
    a.choose_action([], _Frame([_layer(20)], levels=1))
    assert a._reward_src == "grounded:template(win-grid)"


def test_level_states_reset_on_levelup(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_WINREWARD="1")
    _play_level(a)
    assert a._level_states == []


# --- Task 3: rprog acima do 2phase quando a reward foi validada pela vitoria ---
from collections import deque


def _prime_rprog(a, key, deltas):
    a._rprog[key] = deque(deltas, maxlen=10)


def test_rprog_precedes_two_phase_with_win_reward(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_WINREWARD="1")
    _play_level(a)                                        # define _win_reward
    a._win_seq = []      # sem solution-replay (layout sintetico identico o manteria vivo)
    a.choose_action([], _Frame([_layer(20)], levels=1))   # adota a reward win:
    a.action_counter += 1
    a._last_effect_kind = "structural"                    # o 2phase dispararia
    a.choose_action([], _Frame([_layer(16)], levels=1))
    key = a._pending_log["reasoning"]["key"]              # chave presente neste layout
    _prime_rprog(a, key, [0.5, 0.6, 0.7])
    a._last_effect_kind = "structural"
    a.action_counter += 1
    a.choose_action([], _Frame([_layer(16)], levels=1))   # mesmo layout -> mesma chave existe
    assert a._pending_log["reasoning"]["layer"] == "rprog"
    assert a._pending_log["reasoning"]["key"] == key


def test_two_phase_still_first_without_flag(monkeypatch):
    a = _agent(monkeypatch)                                # flag off
    _play_level(a)
    a.choose_action([], _Frame([_layer(20)], levels=1))
    a.action_counter += 1
    a._last_effect_kind = "structural"
    a.choose_action([], _Frame([_layer(16)], levels=1))
    key = a._pending_log["reasoning"]["key"]
    _prime_rprog(a, key, [0.5, 0.6, 0.7])
    a._last_effect_kind = "structural"
    a.action_counter += 1
    a.choose_action([], _Frame([_layer(16)], levels=1))
    assert a._pending_log["reasoning"]["layer"] != "rprog"


def test_rprog_uncapped_ignores_max(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_WINREWARD="1")
    a._rprog_fires = a._rprog_max

    class C:  # candidato minimo
        key = "ACTION6@c5s0@1,1"
    _prime_rprog(a, C.key, [0.5, 0.6, 0.7])
    assert a._rprog_decide([C()]) is None
    assert a._rprog_decide([C()], uncapped=True) == C.key
