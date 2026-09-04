# agents/causal/policy.py
from __future__ import annotations

import random
from collections import namedtuple

from arcengine import GameAction

Candidate = namedtuple("Candidate", "action x y key has_object obj_size", defaults=(0,))

GRID_N = 6


def cell_center(gx: int, gy: int) -> tuple[int, int]:
    x = int((gx + 0.5) * 64 / GRID_N)
    y = int((gy + 0.5) * 64 / GRID_N)
    return x, y


def cell_of(x: int, y: int) -> tuple[int, int]:
    gx = min(GRID_N - 1, int(x) * GRID_N // 64)
    gy = min(GRID_N - 1, int(y) * GRID_N // 64)
    return gx, gy


def _as_action(a):
    # available_actions pode vir como int (id) ou GameAction. Normaliza p/ GameAction.
    return a if isinstance(a, GameAction) else GameAction.from_id(a)


def action_key(action, cell=None) -> str:
    if not action.is_complex():
        return action.name
    if cell is None:
        return f"{action.name}@empty"
    gx, gy = cell
    return f"{action.name}@cell={gx},{gy}"


def _size_bucket(n: int) -> int:
    for i, b in enumerate((4, 16, 64, 256, 1024)):
        if n < b:
            return i
    return 5


def click_key(scene, x: int, y: int) -> str:
    """Chave de clique: (cor, tamanho, celula-espacial). Preserva info espacial pra
    distinguir botoes em posicoes diferentes (fix: vc33 exige clicar botoes distintos),
    mas embute a classe visual (cor+size) pra diagnostico/agrupamento futuro.
    Fundo/vazio -> ACTION6@bg (sem posicao: todos os cliques no fundo sao iguais)."""
    gx, gy = cell_of(x, y)
    for o in scene.objects:
        if (y, x) in o.cells:
            return f"ACTION6@c{o.color}s{_size_bucket(o.size)}@{gx},{gy}"
    return "ACTION6@bg"


def _object_size_at(scene, x: int, y: int) -> int:
    """Tamanho do objeto sob (x=col, y=row). 0 se fundo/vazio."""
    for o in scene.objects:
        if (y, x) in o.cells:
            return o.size
    return 0


def _object_cells(scene) -> set:
    occ = set()
    for o in scene.objects:
        for (r, c) in o.cells:
            occ.add(cell_of(c, r))   # x=col, y=row
    return occ


def candidates(scene, available_actions, clickmap: bool = False,
               click_colors: set | None = None) -> list:
    """Gera candidatos de acao. click_colors: se fornecido, SO gera candidatos de
    clique em objetos dessas cores (game knowledge). None = sem filtro extra."""
    out = []
    occ = _object_cells(scene)
    for a in available_actions:
        action = _as_action(a)
        if not action.is_complex():
            out.append(Candidate(action, None, None, action.name, False))
        elif clickmap:
            # Object-centric: candidate no CENTROID de cada objeto pequeno.
            seen_keys = set()
            for o in scene.objects:
                if o.size > 100:
                    continue          # HUD/parede/fundo: nem gera candidato
                # Game knowledge: se click_colors definido, so clica nessas cores.
                if click_colors is not None and o.color not in click_colors:
                    continue
                x = int(round(o.centroid[1]))   # col
                y = int(round(o.centroid[0]))   # row
                gx, gy = cell_of(x, y)
                key = f"ACTION6@c{o.color}s{_size_bucket(o.size)}@{gx},{gy}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    out.append(Candidate(action, x, y, key, True, o.size))
            # background fallback (centro do grid)
            out.append(Candidate(action, 32, 32, "ACTION6@bg", False, 0))
        else:
            for gy in range(GRID_N):
                for gx in range(GRID_N):
                    x, y = cell_center(gx, gy)
                    sz = _object_size_at(scene, x, y)
                    key = action_key(action, (gx, gy))
                    out.append(Candidate(action, x, y, key, (gx, gy) in occ, sz))
    return out


class Policy:
    def __init__(self, seed: int = 0, epsilon: float = 0.05):
        self._rng = random.Random(seed)
        self.epsilon = epsilon

    def score(self, cand, model, seen_effects, budget_frac, novelty=None, prior=None,
              rprog=None, productive_colors=None) -> float:
        eff, conf = model.predict(cand.key)
        s = 0.0
        if model.is_progress(cand.key):
            s += 10.0 * (1 + (1 - budget_frac))
        if novelty is None:
            if eff is None:
                s += 3.0
            elif conf < 0.8:
                s += 1.5
            if eff is not None and eff.kind not in seen_effects:
                s += 0.5
        else:
            y = novelty.yield_estimate(cand.key)
            ctrl = conf if eff is not None else 1.0
            s += 3.0 * y * ctrl
        if eff is not None and eff.kind == "none":
            s -= 2.0
        if cand.has_object:
            s += 0.5
            # Objetos grandes sao quase sempre inertes (HUD, parede, fundo).
            if cand.obj_size > 100:
                s -= 4.0
        if prior is not None:
            from .transfer import abstract_feature, W_PRIOR
            s += W_PRIOR * prior.productivity(abstract_feature(cand))
        # Bonus por progresso de reward observado (model-free, rprog integrado)
        if rprog is not None:
            row = rprog.get(cand.key)
            if row and len(row) >= 3:
                avg = sum(row) / len(row)
                if avg > 0:
                    s += min(2.0, 5.0 * avg)
        # Transfer entre níveis: cores que resolveram níveis anteriores ganham bonus.
        if productive_colors and "@c" in cand.key:
            try:
                c = int(cand.key.split("@c")[1].split("s")[0])
                if c in productive_colors:
                    s += 2.0
            except (ValueError, IndexError):
                pass
        return s

    def decide(self, scene, model, available_actions, seen_effects, budget_frac,
               novelty=None, prior=None, clickmap=False, rprog=None,
               productive_colors=None, last_key=None):
        cands = candidates(scene, available_actions, clickmap=clickmap)
        if not cands:
            return None
        # Budget awareness: explora mais no inicio, exploita no final
        eps = self.epsilon * max(0.2, budget_frac)
        if self._rng.random() < eps:
            return self._rng.choice(cands)
        best, best_s = None, None
        for c in cands:
            sc = self.score(c, model, seen_effects, budget_frac, novelty=novelty,
                            prior=prior, rprog=rprog, productive_colors=productive_colors)
            # Click diversity: penaliza repetir a mesma acao da rodada anterior
            if last_key and c.key == last_key:
                sc -= 3.0
            if best_s is None or sc > best_s:
                best, best_s = c, sc
        return best
