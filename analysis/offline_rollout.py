# analysis/offline_rollout.py
# §4 — Coleta limpa de dados p/ pré-treino offline (Arcade OFFLINE → .jsonl →
# matriz de transição embarcada). Ferramentas DETERMINÍSTICAS de saneamento:
# NÃO é embarcado no notebook Kaggle (roda no desktop, fora das 9h de GPU).
from __future__ import annotations

from collections import Counter


def is_deadlock(actions, scores=None, window: int = 6, min_repeats: int = 3) -> bool:
    """Episódio preso: a cauda de `window` ações é um ciclo curto repetido
    ≥ min_repeats vezes SEM ganho de score na janela. Ruído estático que
    poluiria a matriz de transição."""
    if len(actions) < window:
        return False
    tail = list(actions[-window:])
    if scores is not None and len(scores) >= window and scores[-1] > scores[-window]:
        return False                       # houve progresso → não é deadlock
    for p in range(1, window // min_repeats + 1):
        reps = window // p
        if reps >= min_repeats and tail[:p] * reps == tail[:p * reps]:
            return True
    return False


def filter_trajectories(episodes, window: int = 6, min_repeats: int = 3):
    """Descarta episódios em deadlock. episode = {'actions': [...], 'scores': [...]?}."""
    return [
        e for e in episodes
        if not is_deadlock(e.get("actions", []), e.get("scores"), window, min_repeats)
    ]


def _pattern_key(t):
    """Normaliza uma transição p/ o par (ação, efeito), ignorando o estado exato
    (deltas absolutos são descartados — a mecânica é abstrata)."""
    if len(t) == 3:            # (sig, key, effect_kind)
        _, key, eff = t
    else:                      # (key, effect_kind)
        key, eff = t
    return (key, eff)


def frequent_patterns(transitions, min_support: int = 3):
    """Padrões de transformação (ação→efeito) que se repetem, ordenados por
    ganho MDL ≈ suporte × |descrição|. Candidatos a pré-config de física."""
    counts = Counter(_pattern_key(t) for t in transitions)
    out = [
        {"pattern": pat, "support": n, "mdl_gain": n * len(str(pat))}
        for pat, n in counts.items() if n >= min_support
    ]
    out.sort(key=lambda d: d["mdl_gain"], reverse=True)
    return out


def mdl_physics(transitions, min_support: int = 3) -> dict:
    """Comprime os padrões frequentes numa pré-config de física serializável:
    ação → efeito modal (só efeitos produtivos, efeito != 'none')."""
    physics = {}
    for p in frequent_patterns(transitions, min_support):
        key, eff = p["pattern"]
        if eff == "none":
            continue
        if key not in physics or p["support"] > physics[key][1]:
            physics[key] = (eff, p["support"])
    return {k: v[0] for k, v in physics.items()}
