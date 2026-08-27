# agents/causal/sandbox.py
# Sandbox p/ código `decide(scene)->action_key` gerado pelo nosso LLM.
# Threat model = acidente/alucinação (não adversário). exec com builtins
# restritos + timeout via thread + captura total → None no fallback.
from __future__ import annotations

import threading

_SAFE_NAMES = (
    "len", "min", "max", "sorted", "range", "abs", "sum", "enumerate",
    "list", "dict", "set", "tuple", "int", "float", "str", "bool",
    "any", "all", "map", "filter", "zip", "round", "frozenset",
)
SAFE_BUILTINS = {n: (__builtins__[n] if isinstance(__builtins__, dict)
                     else getattr(__builtins__, n)) for n in _SAFE_NAMES}
SAFE_BUILTINS["print"] = lambda *a, **k: None      # print no-op


def compile_decide(source):
    if not source or "decide" not in source:
        return None
    g = {"__builtins__": SAFE_BUILTINS}
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


def execute_code_goal(source, scene, timeout: float = 0.5):
    fn = compile_decide(source)
    if fn is None:
        return None
    return run_decide(fn, scene, timeout)
