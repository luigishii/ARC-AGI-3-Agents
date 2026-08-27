# agents/causal/ranker.py
# Rankeia candidatos `decide(scene)` do LLM contra o forward-model aprendido
# (CausalModel/TransitionModel/NoveltyModel) — o análogo AGI-3 de "validar em
# pares de treino", em ms (lookups em dict). Não gasta ação no ambiente.
#
# Camadas de filtragem determinística (backlog §1):
#  - retrodiction: o candidato tem que "retrodizer" o histórico gravado
#    (transition_buffer). Se erra o passado imediato, não merece o presente.
#  - trava de available_actions: descarta candidato que produz ação indisponível.
#  - deduplicação semântica: dois candidatos que geram a MESMA ação = simula 1.
#  - penalização de no-op: ação sabidamente sem efeito → score prospectivo 0.
from __future__ import annotations

from .sandbox import execute_code_goal
from .novelty import state_signature
from .dsl import DSL

W_RETRO = 2.0        # peso do termo de retrodição (primário, acima da novidade)


def _avail_names(available):
    """Normaliza available_actions (GameAction|int|str) p/ conjunto de nomes."""
    if available is None:
        return None
    out = set()
    for a in available:
        name = getattr(a, "name", None)
        if name is None:
            try:
                from arcengine import GameAction
                name = GameAction.from_id(a).name
            except Exception:
                name = str(a)
        out.add(name)
    return out


def _base_action(key: str) -> str:
    """Nome da ação-base de um action_key ('ACTION6@cell=2,3' -> 'ACTION6')."""
    return key.split("@", 1)[0]


def retrodiction_score(source, buffer, extra) -> float:
    """Fração líquida de transições gravadas que o candidato retrodiz.

    Para cada (scene_t, key_taken_t, effect_kind_t) do buffer, re-executa o
    candidato sobre scene_t. Se ele reproduz a ação realmente tomada, isso é
    uma "concordância": conta a favor se aquela ação foi produtiva (efeito !=
    'none') e contra se foi morta. Sem sobreposição com o histórico → 0.0
    (candidato inédito não é punido). Determinístico, µs.
    """
    if not buffer:
        return 0.0
    prod = dead = 0
    for scene_t, key_taken, eff_kind in buffer:
        pred = execute_code_goal(source, scene_t, extra=extra)
        if pred == key_taken:
            if eff_kind == "none":
                dead += 1
            else:
                prod += 1
    return (prod - dead) / len(buffer)


def rank_candidates(sources, scene, model, tmodel, novelty, moves,
                    buffer=None, available=None):
    extra = {**DSL, "MOVES": moves}
    sig = state_signature(scene)
    avail = _avail_names(available)
    seen_keys = set()
    best_src, best_score = None, None
    for src in sources:
        key = execute_code_goal(src, scene, extra=extra)
        if not isinstance(key, str):
            continue
        # trava de available_actions: ação-base tem que estar disponível agora
        if avail is not None and _base_action(key) not in avail:
            continue
        # deduplicação semântica: mesma ação resultante = simula só a 1ª
        if key in seen_keys:
            continue
        seen_keys.add(key)

        eff, conf = model.predict(key)
        nxt = tmodel.predict_next(sig, key)
        ctrl = conf if eff is not None else 1.0
        nov = novelty.novelty(nxt) if nxt is not None else 1.0   # inédito → fronteira
        # penalização por no-op: ação conhecida como sem efeito / estado imutável
        no_op = (eff is not None and eff.kind == "none") or (nxt is not None and nxt == sig)
        prospective = 0.0 if no_op else nov * ctrl

        retro = retrodiction_score(src, buffer, extra) if buffer else 0.0
        score = W_RETRO * retro + prospective
        if best_score is None or score > best_score:
            best_score, best_src = score, src
    return best_src
