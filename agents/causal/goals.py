# agents/causal/goals.py
# A: predicado de meta / reward_function sintetizada pelo LLM (contrato OPINE:
# reward_function(state) -> (reward, goal_flag)). Fecha as 4 funções do game_engine
# e torna o IW GOAL-DIRECTED (planeja rumo à meta, não só a novidade).
# Check estático anti-trapaça: rejeita predicado que reconhece a meta SEM olhar o
# state (ex: lê estado cacheado de nível posterior).
from __future__ import annotations

from .typed_model import compile_rule

# tokens que denunciam trapaça / estado escondido (não são necessários num predicado puro)
_FORBIDDEN = ("global ", "nonlocal ", "import ", "__")


def compile_reward(source):
    """Compila reward_function(state) sob builtins restritos. Falha → None."""
    return compile_rule(source, name="reward_function")


def static_reward_check(source) -> bool:
    """Aceita só se: compila, recebe `state`, USA o `state` no corpo (não só na
    assinatura → senão reconhece a meta sem olhar a mecânica) e não referencia
    estado global/escondido."""
    if not source or "reward_function" not in source:
        return False
    if any(tok in source for tok in _FORBIDDEN):
        return False
    fn = compile_reward(source)
    if fn is None or fn.__code__.co_argcount < 1:
        return False
    param = fn.__code__.co_varnames[0]
    return source.count(param) >= 2      # aparece na assinatura E no corpo


def goal_fn_from_reward(reward_fn):
    """Adapta reward_function -> goal_fn(state)->bool p/ o IW. Lê o goal_flag
    (2º elemento) e é à prova de exceção (falha → não-goal)."""
    def goal(state):
        try:
            r = reward_fn(state)
            if isinstance(r, (tuple, list)) and len(r) >= 2:
                return bool(r[1])
            return bool(r)
        except Exception:
            return False
    return goal


def value_fn_from_reward(reward_fn):
    """Adapta reward_function -> value(state)->float p/ o IW best-first. Lê o reward
    (1º elemento) e é à prova de exceção (falha → -inf, nunca escolhido)."""
    def value(state):
        try:
            r = reward_fn(state)
            if isinstance(r, (tuple, list)) and len(r) >= 1:
                return float(r[0])
            return float(r)
        except Exception:
            return float("-inf")
    return value


def grounded_reward_fn(avatar_color, target_color):
    """Reward GROUNDED (calculada, NÃO chutada pelo LLM): -manhattan(avatar, alvo),
    ancorada por COR (estável entre frames, ao contrário de índice). goal_flag quando
    distância==0. Exception-safe (entrada ruim -> (0,False)). Lição do docs/GAMES.md:
    a vitória é sempre espacial/grounded, nunca contagem-de-cor."""
    def reward_function(state):
        try:
            objs = [a for _, a in state]
            av = [o for o in objs if o.get("color") == avatar_color]
            tg = [o for o in objs if o.get("color") == target_color]
            if not av or not tg:
                return (0.0, False)
            a, t = av[0], tg[0]
            d = abs(a["x"] - t["x"]) + abs(a["y"] - t["y"])
            return (-float(d), d == 0)
        except Exception:
            return (0.0, False)
    return reward_function


def grounded_multi_reward_fn(max_size=64):
    """Reward por-classe p/ ALINHAMENTO/POSICIONAMENTO (sokoban/manipulacao, docs/GAMES.md):
    soma da menor distancia par-a-par entre objetos PEQUENOS de MESMA cor (marcador<->alvo,
    caixa<->destino). Exclui blocos grandes (HUD/parede/fundo, size>max_size) — o erro que
    a reward chutada pelo LLM cometia. goal_flag quando tudo coincide (soma==0). Safe."""
    def reward_function(state):
        try:
            byc = {}
            for _, o in state:
                if o.get("size", 10 ** 9) <= max_size:
                    byc.setdefault(o["color"], []).append(o)
            total = 0
            for objs in byc.values():
                if len(objs) >= 2:
                    total += min(abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
                                 for i, a in enumerate(objs) for b in objs[i + 1:])
            return (-float(total), total == 0)
        except Exception:
            return (0.0, False)
    return reward_function


def grounded_pattern_reward_fn(split=32, max_size=64):
    """Reward por-classe p/ PINTURA/SEQUENCIA/FLUXO (docs/GAMES.md): esses jogos tem uma
    REFERENCIA (tipicamente no topo) e uma area EDITAVEL (embaixo); vence quando casam.
    Proxy grounded: compara a composicao de cor da metade de cima vs baixo (objetos
    pequenos) -> diferenca simetrica dos multiconjuntos; 0 = casam. goal_flag em 0.
    Proxy (nao deteccao exata da referencia), exception-safe."""
    from collections import Counter

    def reward_function(state):
        try:
            top, bot = Counter(), Counter()
            for _, o in state:
                if o.get("size", 10 ** 9) <= max_size:
                    (top if o["y"] < split else bot)[o["color"]] += 1
            if not top or not bot:
                return (0.0, False)
            diff = sum((top - bot).values()) + sum((bot - top).values())
            return (-float(diff), diff == 0)
        except Exception:
            return (0.0, False)
    return reward_function


def grounded_pair_reward_fn(max_size=64):
    """Reward por-classe p/ jogos de PARES/MIRROR (m0r0, ls20): soma das distancias
    minimas entre objetos PEQUENOS de cores DIFERENTES agrupados por proximidade.
    Complementa multi_reward (mesma cor) com logica de convergencia de pares distintos.
    goal_flag quando total de pares <= 1. Safe."""
    def reward_function(state):
        try:
            pts = [(o["x"], o["y"], o["color"], o.get("shape", 0))
                   for _, o in state if o.get("size", 10 ** 9) <= max_size]
            if len(pts) < 2:
                return (0.0, False)
            # Para pares mirror/shape: minimiza soma de distancias entre objetos
            # de mesmo shape (forma) mas cor diferente → convergencia de pares
            by_shape = {}
            for x, y, c, s in pts:
                by_shape.setdefault(s, []).append((x, y, c))
            total = 0
            for objs in by_shape.values():
                if len(objs) >= 2:
                    total += min(abs(a[0] - b[0]) + abs(a[1] - b[1])
                                 for i, a in enumerate(objs) for b in objs[i + 1:])
            return (-float(total), total == 0)
        except Exception:
            return (0.0, False)
    return reward_function


def grounded_count_reward_fn(max_size=64):
    """Reward por-classe p/ SOKOBAN/PEG (docs/GAMES.md): conta objetos pequenos —
    menos objetos = melhor (pegs removidos) ou mais objetos numa zona = melhor
    (caixas no alvo). Proxy: -num_objetos_pequenos. goal_flag quando <= 2. Safe."""
    def reward_function(state):
        try:
            n = sum(1 for _, o in state if o.get("size", 10 ** 9) <= max_size)
            return (-float(n), n <= 2)
        except Exception:
            return (0.0, False)
    return reward_function


def accept_reward(source, sample_states, min_states=3):
    """Aceitação COMPORTAMENTAL da reward: avalia em estados reais e rejeita patológicas.
    Retorna (aceito, motivo). Cold-start: < min_states estados -> aceita (bootstrap)."""
    fn = compile_reward(source)
    if fn is None:
        return (False, "não compila")
    if len(sample_states) < min_states:
        return (True, "poucos estados p/ julgar (cold-start)")
    vals, flags = [], []
    for st in sample_states:
        try:
            r = fn(st)
        except Exception:
            return (False, "levanta exceção em estado real")
        if isinstance(r, (tuple, list)) and len(r) >= 2:
            vals.append(float(r[0])); flags.append(bool(r[1]))
        else:
            vals.append(float(r)); flags.append(bool(r))
    if all(flags):
        return (False, "goal_flag=True em TODO estado (falso-positivo)")
    distinct_states = len({repr(st) for st in sample_states}) > 1
    if distinct_states and len({round(v, 6) for v in vals}) <= 1:
        return (False, "reward escalar CONSTANTE entre estados distintos (sem gradiente)")
    return (True, "ok")
