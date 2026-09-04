"""Fixes dos 3 bugs isolados no run local dos 25 jogos em modo cego (04/set).

Evidencia que motivou cada teste:
  A) reward ancorava na COR 0 (fundo/vazio) em 5 de 25 jogos -> -dist(cor0->cor8)
     em cn04/cd82/tr87 (cor 0 como avatar) e ls20/sp80 (cor 0 como alvo).
  B) early-exit matava o jogo em 65% do budget mesmo ainda descobrindo: tn36
     cruzou o nivel na acao 88 num run, e morreu na 52 quando o budget foi 80.
  C) early-exit agressivo matou o ft09 na acao 24 -- antes de completar uma
     varredura de cliques (36 celulas candidatas).
"""
from types import SimpleNamespace as NS

from agents.causal.agent import (
    CYCLE_MAX_KEYS,
    CYCLE_WINDOW,
    DISCOVERY_PATIENCE,
    MIN_SWEEP_ACTIONS,
    UNKNOWN_BUDGET,
    CausalObjectAgent,
    _background_colors,
    _pick_target,
)
from agents.causal.goals import grounded_reward_fn


def _obj(color, size, bbox, centroid):
    return NS(color=color, size=size, bbox=bbox, centroid=centroid)


def _agent(**over):
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a._cleanup = False
    a._init_causal_state()
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _frame(levels=0):
    return NS(state=None, levels_completed=levels)


# ---------------------------------------------------------------- A: fundo/vazio
def test_background_colors_pega_dominante_e_vazio():
    """Fundo = cor com mais pixels; cor 0 (vazio) sempre conta como fundo."""
    scene = NS(objects=[
        _obj(3, 900, (0, 0, 40, 40), (20, 20)),   # dominante
        _obj(0, 3, (5, 5, 5, 7), (5, 6)),         # buraco no fundo
        _obj(9, 15, (1, 1, 3, 5), (2, 3)),
    ])
    bg = _background_colors(scene)
    assert 3 in bg and 0 in bg and 9 not in bg


def test_pick_target_nao_escolhe_cor_de_fundo_nem_ruido():
    """Reproduz o ls20: cor 0 com 3px era a 'mais rara' e vencia o alvo real."""
    objs = [
        _obj(9, 15, (0, 0, 2, 4), (1, 2)),          # 0 avatar
        _obj(3, 900, (10, 10, 40, 40), (25, 25)),   # 1 fundo dominante
        _obj(0, 3, (32, 20, 32, 22), (32, 21)),     # 2 ruido cor-vazio  <- era escolhido
        _obj(5, 22, (40, 30, 44, 34), (42, 32)),    # 3 alvo plausivel
        _obj(5, 25, (50, 30, 54, 34), (52, 32)),
    ]
    bg = _background_colors(NS(objects=objs))
    t = _pick_target(objs, 0, limit=None, bg_colors=bg)
    assert t is not None and objs[t].color == 5


def test_pick_target_cai_de_volta_quando_tudo_seria_excluido():
    """Nunca fica pior que hoje: se a exclusao esvazia, mantem o candidato antigo."""
    objs = [
        _obj(9, 15, (0, 0, 2, 4), (1, 2)),         # avatar
        _obj(3, 900, (10, 10, 40, 40), (25, 25)),  # fundo
        _obj(0, 2, (5, 5, 5, 6), (5, 5)),          # so sobra ruido de fundo
    ]
    bg = _background_colors(NS(objects=objs))
    assert _pick_target(objs, 0, limit=None, bg_colors=bg) is not None


def test_grounded_reward_desambigua_avatar_por_tamanho():
    """ls20 tem 5 objetos cor 9; ancorar em av[0] pegava a peca errada."""
    state = [
        ("h1", {"x": 36, "y": 12, "color": 9, "size": 5}),    # peca errada
        ("h2", {"x": 36, "y": 43, "color": 9, "size": 15}),   # avatar real
        ("h3", {"x": 40, "y": 43, "color": 5, "size": 22}),   # alvo
    ]
    fn = grounded_reward_fn(9, 5, avatar_size=15)
    r, _ = fn(state)
    assert r == -4.0        # |36-40| + |43-43|, medido a partir do avatar de size 15


# ---------------------------------------------------------------- B/C: budget
def test_jogo_desconhecido_ganha_budget_maior(monkeypatch):
    monkeypatch.delenv("CAUSAL_MAX_ACTIONS", raising=False)
    assert _agent().MAX_ACTIONS == UNKNOWN_BUDGET
    assert UNKNOWN_BUDGET > 80


def test_nao_desiste_enquanto_ainda_descobre():
    """tn36 cruzou na acao 88; o corte em 65% do budget matava antes disso."""
    a = _agent(MAX_ACTIONS=100, action_counter=70, _seen_effects={"structural"})
    a._last_discovery = 70            # descobriu agora
    assert a.is_done([], _frame()) is False


def test_desiste_quando_para_de_descobrir():
    a = _agent(MAX_ACTIONS=100, action_counter=70, _seen_effects={"structural"})
    a._last_discovery = 70 - DISCOVERY_PATIENCE
    assert a.is_done([], _frame()) is True


def test_exit_agressivo_espera_uma_varredura():
    """ft09 morreu na acao 24 com 36 celulas de clique candidatas."""
    a = _agent(MAX_ACTIONS=80, action_counter=24, _seen_effects={"none"})
    a._last_discovery = 0
    assert a.is_done([], _frame()) is False
    a.action_counter = MIN_SWEEP_ACTIONS
    assert a.is_done([], _frame()) is True


# ------------------------------------------------- D: ciclo curto pos-level-up
def _fix_agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 200
    a._init_causal_state()
    return a


def test_antifix_quebra_ciclo_de_4_chaves(monkeypatch):
    """vc33 cruzou o L1 na acao 37 e gastou 119 das 164 acoes restantes girando
    entre 4 botoes (c9s2 nas celulas 0,1..0,4). Repeticao (AAA) e oscilacao (ABAB)
    nao pegam um ciclo de periodo 4."""
    from agents.causal.policy import Candidate
    a = _fix_agent(monkeypatch, CAUSAL_FIX="1")
    cands = [Candidate(None, None, None, k, False) for k in ("A", "B", "C", "D", "E")]
    km = {c.key: c for c in cands}
    out = None
    for i in range(CYCLE_WINDOW):
        k = "ABCD"[i % 4]
        out = a._antifix(km[k], cands, km)
    assert out.key == "E", out.key      # forcado para fora do ciclo
    assert a._fix_breaks >= 1


def test_alvo_nunca_indistinguivel_do_avatar():
    """ka59 saiu com -dist(cor14#8->cor14#8): mesma cor E mesmo tamanho, entao a
    reward ancora no MESMO objeto dos dois lados -> distancia 0 sempre ->
    goal_flag True em 185 de 198 estados reais."""
    objs = [
        _obj(14, 8, (0, 0, 2, 3), (1, 1)),        # 0 avatar
        _obj(14, 8, (20, 20, 22, 23), (21, 21)),  # 1 gemeo: indistinguivel
        _obj(7, 6, (40, 40, 42, 42), (41, 41)),   # 2 alvo utilizavel
        _obj(3, 900, (0, 0, 40, 40), (25, 25)),   # 3 fundo
    ]
    bg = _background_colors(NS(objects=objs))
    t = _pick_target(objs, 0, limit=None, bg_colors=bg)
    assert t is not None
    assert not (objs[t].color == 14 and objs[t].size == 8)


def test_alvo_nao_e_estrutura_gigante():
    """m0r0 mirava um objeto de 1294px (49% da area): metade do tabuleiro, nao alvo.
    O piso pegava ruido embaixo; faltava teto em cima."""
    objs = [
        _obj(10, 25, (0, 0, 4, 4), (2, 2)),          # 0 avatar (marcador)
        _obj(12, 1299, (10, 0, 40, 40), (25, 20)),   # 1 metade do tabuleiro
        _obj(11, 1294, (10, 41, 40, 63), (25, 50)),  # 2 outra metade  <- era escolhido
        _obj(10, 25, (50, 50, 54, 54), (52, 52)),    # 3 marcador irmao
    ]
    bg = _background_colors(NS(objects=objs))
    t = _pick_target(objs, 0, limit=None, bg_colors=bg)
    assert t is not None and objs[t].size <= 25


def test_sem_alvo_quando_so_sobra_estrutura():
    """m0r0 e um jogo de espelho: os dois marcadores co-movem, entao o irmao sai como
    parte do avatar e sobram so as duas metades do tabuleiro (49% da area cada).
    Melhor ficar SEM reward de navegacao do que mirar a estrutura."""
    objs = [
        _obj(10, 25, (0, 0, 4, 4), (2, 2)),          # 0 avatar
        _obj(12, 1299, (10, 0, 40, 40), (25, 20)),   # 1 metade do tabuleiro
        _obj(11, 1294, (10, 41, 40, 63), (25, 50)),  # 2 outra metade
        _obj(10, 25, (50, 50, 54, 54), (52, 52)),    # 3 irmao co-movido
    ]
    objs[3].id = 77
    bg = _background_colors(NS(objects=objs))
    assert _pick_target(objs, 0, limit=None, exclude_ids={77}, bg_colors=bg) is None


def test_accept_reward_fn_rejeita_falso_positivo():
    """A trava comportamental existe mas so era aplicada na reward do LLM."""
    from agents.causal.goals import accept_reward_fn
    estados = [
        [("h", {"x": 1, "y": 1, "color": 4, "size": 3})],
        [("h", {"x": 9, "y": 9, "color": 4, "size": 3})],
        [("h", {"x": 5, "y": 2, "color": 4, "size": 3})],
    ]
    ok, motivo = accept_reward_fn(lambda st: (0.0, True), estados)
    assert ok is False and "falso-positivo" in motivo
    ok2, _ = accept_reward_fn(lambda st: (-float(st[0][1]["x"]), False), estados)
    assert ok2 is True


def test_antifix_nao_quebra_com_variedade(monkeypatch):
    """Janela cheia mas com mais de CYCLE_MAX_KEYS chaves distintas = exploracao
    saudavel, nao ciclo: nao deve sobrepor."""
    from agents.causal.policy import Candidate
    a = _fix_agent(monkeypatch, CAUSAL_FIX="1")
    keys = [f"K{i}" for i in range(CYCLE_MAX_KEYS + 2)]
    cands = [Candidate(None, None, None, k, False) for k in keys]
    km = {c.key: c for c in cands}
    for i in range(CYCLE_WINDOW):
        out = a._antifix(km[keys[i % len(keys)]], cands, km)
    assert a._fix_breaks == 0
    assert out.key in keys
