# agents/causal/ranker.py
# Rankeia candidatos `decide(scene)` do LLM contra o forward-model aprendido
# (CausalModel/TransitionModel/NoveltyModel) — o análogo AGI-3 de "validar em
# pares de treino", em ms (lookups em dict). Não gasta ação no ambiente.
from __future__ import annotations

from .sandbox import execute_code_goal
from .novelty import state_signature
from .dsl import DSL


def rank_candidates(sources, scene, model, tmodel, novelty, moves):
    extra = {**DSL, "MOVES": moves}
    sig = state_signature(scene)
    best_src, best_score = None, None
    for src in sources:
        key = execute_code_goal(src, scene, extra=extra)
        if not isinstance(key, str):
            continue
        eff, conf = model.predict(key)
        nxt = tmodel.predict_next(sig, key)
        ctrl = conf if eff is not None else 1.0
        nov = novelty.novelty(nxt) if nxt is not None else 1.0   # inédito → fronteira
        score = nov * ctrl
        if best_score is None or score > best_score:
            best_score, best_src = score, src
    return best_src
