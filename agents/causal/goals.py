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
