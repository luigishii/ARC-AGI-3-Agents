# agents/causal/sandbox.py
# Sandbox p/ código `decide(scene)->action_key` gerado pelo nosso LLM.
# Threat model = acidente/alucinação (não adversário). exec com builtins
# restritos + timeout via thread + captura total → None no fallback.
from __future__ import annotations

import re
import threading

_SAFE_NAMES = (
    "len", "min", "max", "sorted", "range", "abs", "sum", "enumerate",
    "list", "dict", "set", "tuple", "int", "float", "str", "bool",
    "any", "all", "map", "filter", "zip", "round", "frozenset",
)
SAFE_BUILTINS = {n: (__builtins__[n] if isinstance(__builtins__, dict)
                     else getattr(__builtins__, n)) for n in _SAFE_NAMES}
SAFE_BUILTINS["print"] = lambda *a, **k: None      # print no-op


def compile_decide(source, extra=None):
    if not source or "decide" not in source:
        return None
    g = {"__builtins__": SAFE_BUILTINS}
    if extra:
        g.update(extra)
    try:
        exec(source, g)
    except Exception:
        return None
    fn = g.get("decide")
    return fn if callable(fn) else None


def run_decide(fn, scene, timeout: float = 0.5):
    box = {"val": None}

    def worker():
        try:
            r = fn(scene)
            if isinstance(r, str):
                box["val"] = r
        except Exception:
            box["val"] = None

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None      # hang → abandona a daemon-thread, fallback
    return box["val"]


def execute_code_goal(source, scene, timeout: float = 0.5, extra=None):
    fn = compile_decide(source, extra)
    if fn is None:
        return None
    return run_decide(fn, scene, timeout)


# --- §3: taxonomia de erros por regex (prep p/ self-debug barato) ---
# Classifica o traceback ANTES de gastar token do LLM: só erros semânticos
# escalam; o resto casa num slug determinístico.
_ERR_PATTERNS = (
    ("syntax_error", r"SyntaxError|invalid syntax|unexpected EOF|IndentationError"),
    ("index_oob", r"IndexError|index out of range|out of (?:range|bounds)|KeyError"),
    ("shape_mismatch",
     r"could not broadcast|operands could not|number of dimensions|shape|dimension"),
)


def classify_error(err) -> str:
    """Slug determinístico p/ uma exceção/traceback ('other' = semântico → escala)."""
    if err is None:
        return "none"
    text = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    for slug, pat in _ERR_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return slug
    return "other"


def run_decide_diag(fn, scene, timeout: float = 0.5):
    """Como run_decide, mas devolve (key|None, slug|None) com a causa da falha.
    Sucesso → (key, None); exceção → (None, slug); rodou mas sem ação válida →
    (None, 'no_op'); travou → (None, 'infinite_loop')."""
    box = {"val": None, "err": None}

    def worker():
        try:
            r = fn(scene)
            if isinstance(r, str):
                box["val"] = r
            else:
                box["err"] = "no_op"       # executou mas não produziu ação
        except Exception as e:             # noqa: BLE001 — captura total por design
            box["err"] = classify_error(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return (None, "infinite_loop")
    return (box["val"], box["err"])


def execute_code_goal_diag(source, scene, timeout: float = 0.5, extra=None):
    """execute_code_goal + slug de diagnóstico (não compila → 'syntax_error')."""
    fn = compile_decide(source, extra)
    if fn is None:
        return (None, "syntax_error")
    return run_decide_diag(fn, scene, timeout)


def execute_code_goal_verified(source, scene, timeout: float = 0.5, extra=None):
    """Double-eval determinístico (OPINE-World "achado do dia"): roda o decide
    2× e só aceita se as saídas coincidirem. Candidato com estado escondido /
    não-determinístico → None (rejeitado) — o planner forward assume transição
    pura e determinística."""
    fn = compile_decide(source, extra)
    if fn is None:
        return None
    a = run_decide(fn, scene, timeout)
    b = run_decide(fn, scene, timeout)
    return a if (a is not None and a == b) else None
