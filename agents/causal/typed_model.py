# agents/causal/typed_model.py
# Componente B do redesign Fase-2 (OPINE-World): world-model FATORADO POR TIPO.
# f_τ: O_τ × A × X → O_τ — cada tipo mecânico tem sua própria regra de transição,
# ACEITA por replay exato SÓ das transições daquele tipo (+ double-eval determinístico).
# Adaptação-chave pro Qwen 7B: valida 1 tipo por vez (vários problemas fáceis) em vez
# do jogo inteiro (1 problema impossível). NumPy/stdlib, offline.
from __future__ import annotations

import threading

from .sandbox import SAFE_BUILTINS

_FAIL = object()      # sentinela: exceção ou timeout na aplicação da regra


def compile_rule(source, name: str = "transition"):
    """Compila o source de uma regra f_τ do LLM num callable `name(obj,action,ctx)`
    sob builtins restritos (reusa o perímetro do sandbox). Falha → None."""
    if not source or name not in source:
        return None
    g = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(source, g)
    except Exception:
        return None
    fn = g.get(name)
    return fn if callable(fn) else None


def _apply(fn, before, action, context, timeout: float):
    """Aplica fn(before,action,context) com timeout via thread; falha → _FAIL."""
    box = {"val": _FAIL}

    def worker():
        try:
            box["val"] = fn(before, action, context)
        except Exception:
            box["val"] = _FAIL

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return _FAIL
    return box["val"]


def verify_transition_fn(fn, transitions, timeout: float = 0.5) -> bool:
    """Aceita a regra sse ela reproduz TODA transição do tipo por igualdade exata,
    determinística (roda 2× e exige saídas iguais). 1 mismatch/nondet/erro rejeita."""
    if fn is None:
        return False
    for t in transitions:
        before, action = t["before"], t["action"]
        ctx, after = t.get("context", {}), t["after"]
        a = _apply(fn, before, action, ctx, timeout)
        b = _apply(fn, before, action, ctx, timeout)
        if a is _FAIL or b is _FAIL:
            return False
        if a != b:               # estado escondido / não-determinismo
            return False
        if a != after:           # replay exato falhou
            return False
    return True


def accept_rule(source, transitions, timeout: float = 0.5, name: str = "transition") -> bool:
    """Compila a regra f_τ do LLM e a aceita sse passa o replay exato das transições
    daquele tipo (verify_transition_fn)."""
    fn = compile_rule(source, name)
    if fn is None:
        return False
    return verify_transition_fn(fn, transitions, timeout)


class TypedWorldModel:
    """Coleção de regras aceitas, uma por tipo. predict() monta o próximo estado
    aplicando f_τ objeto-a-objeto (tipo desconhecido → inalterado). Serializável
    pelos sources (embarca no notebook como os outros modelos)."""

    def __init__(self):
        self.sources = {}        # τ -> source str (aceito)
        self._compiled = {}      # τ -> callable

    def set_rule(self, tau, source) -> None:
        fn = compile_rule(source)
        if fn is None:
            raise ValueError(f"regra inválida p/ tipo {tau!r}")
        self.sources[tau] = source
        self._compiled[tau] = fn

    def predict(self, objects, action, context_fn=None, timeout: float = 0.5):
        """objects = [(τ, obj_dict), ...] → [(τ, next_obj_dict), ...]."""
        out = []
        for tau, obj in objects:
            fn = self._compiled.get(tau)
            if fn is None:
                out.append((tau, obj))
                continue
            ctx = context_fn(obj) if context_fn is not None else {}
            r = _apply(fn, obj, action, ctx, timeout)
            out.append((tau, obj if r is _FAIL else r))
        return out

    def to_dict(self) -> dict:
        return {"sources": dict(self.sources)}

    @classmethod
    def from_dict(cls, d: dict) -> "TypedWorldModel":
        m = cls()
        for tau, src in d.get("sources", {}).items():
            m.set_rule(tau, src)
        return m


# --- Contrato das 4 funções (OPINE game_engine.py) que o Sandbox/prompt impõem ---
REQUIRED_ENGINE_FNS = ("transition_function", "reward_function", "extract_objects")
OPTIONAL_ENGINE_FNS = ("planner",)


def validate_engine_contract(source) -> dict:
    """Compila o source e reporta quais funções do contrato existem e são callable."""
    g = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(source, g)
    except Exception:
        g = {}
    return {name: callable(g.get(name))
            for name in REQUIRED_ENGINE_FNS + OPTIONAL_ENGINE_FNS}


def has_engine_contract(source) -> bool:
    """True sse as 3 funções obrigatórias do contrato estão presentes (planner é opcional)."""
    v = validate_engine_contract(source)
    return all(v[n] for n in REQUIRED_ENGINE_FNS)
