# agents/causal/iw.py
# Iterated Width (IW, Lipovetzky-Geffner) sobre o TypedWorldModel — planner
# blind width-based, CPU puro, ZERO LLM. Substitui o MCTS no espaço de código:
# em grid determinístico com objetos, IW(1)/IW(2) explora ESTRUTURALMENTE (poda por
# novidade de tuplas de átomos) e acha plano onde o MCTS ainda faz rollouts aleatórios.
from __future__ import annotations

from collections import deque

_ATTRS = ("x", "y", "color")


def atoms(state) -> set:
    """Átomos do estado = (índice_do_objeto, atributo, valor). O 'width' do IW é
    sobre tuplas desses átomos."""
    out = set()
    for i, (_tau, obj) in enumerate(state):
        for k in _ATTRS:
            if k in obj:
                out.add((i, k, obj[k]))
    return out


def _tuples(state, width):
    ats = sorted(atoms(state))
    if width <= 1:
        return set(ats)
    pairs = set()
    for i in range(len(ats)):
        for j in range(i, len(ats)):
            pairs.add((ats[i], ats[j]))
    return pairs


def _register(state, seen, width) -> bool:
    """Registra as tuplas do estado; retorna True se ao menos uma era inédita (novo)."""
    ts = _tuples(state, width)
    novel = any(t not in seen for t in ts)
    seen |= ts
    return novel


def _new_count(state, seen, width) -> int:
    return sum(1 for t in _tuples(state, width) if t not in seen)


def iw_search(start, actions, model, goal_fn=None, width=1, max_nodes=1000):
    """BFS com poda por novidade de largura `width`. Com goal_fn → devolve a 1ª ação
    do caminho que atinge o goal (senão None). Sem goal_fn (exploração) → a ação-raiz
    cujo próximo estado revela mais átomos inéditos (None se nenhuma muda nada)."""
    if goal_fn is not None and goal_fn(start):
        return None
    seen = set()
    _register(start, seen, width)

    q = deque()
    best_action, best_new = None, 0
    for a in actions:
        nxt = model.predict(start, a)
        if goal_fn is None:
            n = _new_count(nxt, seen, width)
            if n > best_new:
                best_new, best_action = n, a
        q.append((nxt, a))

    nodes = 0
    while q and nodes < max_nodes:
        state, first = q.popleft()
        nodes += 1
        if goal_fn is not None and goal_fn(state):
            return first
        if not _register(state, seen, width):
            continue                       # não-novo → poda (coração do IW)
        for a in actions:
            q.append((model.predict(state, a), a))

    return None if goal_fn is not None else best_action


def iw_plan(start, actions, model, goal_fn=None, max_width=2, max_nodes=1000):
    """IW iterado: tenta largura 1, depois 2, ... até max_width. 1ª que resolve vence."""
    for w in range(1, max_width + 1):
        r = iw_search(start, actions, model, goal_fn, w, max_nodes)
        if r is not None:
            return r
    return None
