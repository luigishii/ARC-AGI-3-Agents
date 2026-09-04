# agents/causal/navigate.py
from __future__ import annotations

from collections import Counter


def _moved_object(prev, curr):
    # isola o mover RÍGIDO (forma+tamanho preservados = avatar transladando); ignora a barra
    # de HUD que encolhe (muda size/shape) — senão >1 objeto "move" e o avatar nunca é aprendido.
    prevmap = {o.id: o for o in prev.objects}
    rigid = []
    for o in curr.objects:
        po = prevmap.get(o.id)
        if po is None:
            continue
        dr = round(o.centroid[0] - po.centroid[0])
        dc = round(o.centroid[1] - po.centroid[1])
        if (dr, dc) == (0, 0):
            continue
        if o.shape_hash == po.shape_hash and o.size == po.size:
            rigid.append((o.id, (dr, dc)))
    return rigid[0] if len(rigid) == 1 else None


class MovementModel:
    def __init__(self):
        self.vec = {}            # action_key -> {(dr,dc) -> count}
        self.avatar_counts = {}  # obj_id -> count
        self.blocked = {}        # action_key -> count de vezes que NAO moveu o avatar

    def observe(self, key, prev, curr) -> None:
        m = _moved_object(prev, curr)
        if m is None:
            # Se a key ja tinha vetor aprendido mas nao moveu → parede/bloqueio
            if key in self.vec and "@" not in key:
                self.blocked[key] = self.blocked.get(key, 0) + 1
            return
        oid, v = m
        self.vec.setdefault(key, {})
        self.vec[key][v] = self.vec[key].get(v, 0) + 1
        self.avatar_counts[oid] = self.avatar_counts.get(oid, 0) + 1
        # Movimento bem-sucedido: reseta contador de bloqueio
        self.blocked.pop(key, None)

    def move_vector(self, key):
        d = self.vec.get(key)
        return max(d.items(), key=lambda kv: kv[1])[0] if d else None

    def moves(self):
        return {k: self.move_vector(k) for k in self.vec if "@" not in k}

    def avatar_id(self):
        if not self.avatar_counts:
            return None
        return max(self.avatar_counts.items(), key=lambda kv: kv[1])[0]

    def to_dict(self) -> dict:
        return {
            "vec": {k: {f"{dr},{dc}": n for (dr, dc), n in d.items()}
                    for k, d in self.vec.items()},
            "avatar_counts": {str(o): n for o, n in self.avatar_counts.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MovementModel":
        m = cls()
        for k, dd in d.get("vec", {}).items():
            m.vec[k] = {}
            for s, n in dd.items():
                dr, dc = s.split(",")
                m.vec[k][(int(dr), int(dc))] = n
        m.avatar_counts = {int(o): n for o, n in d.get("avatar_counts", {}).items()}
        return m


def _pick_target(others, avatar, freq, target_color=None):
    """Seleciona alvo: se target_color conhecido, filtra por cor. Senao, cor mais rara."""
    ay, ax = avatar.centroid
    if target_color is not None:
        colored = [o for o in others if o.color == target_color]
        if colored:
            return min(colored, key=lambda o: abs(o.centroid[0] - ay) + abs(o.centroid[1] - ax))
    return min(others, key=lambda o: (freq[o.color],
                                      abs(o.centroid[0] - ay) + abs(o.centroid[1] - ax)))


def navigate(scene, move, reached_ids=None, target_color=None):
    """Navega o avatar ate o alvo mais proximo.
    target_color: se fornecido (game knowledge), filtra alvos por essa cor.
    reached_ids: set de object IDs ja alcancados → exclui da selecao de alvo."""
    moves = move.moves()
    if not moves:
        return None
    aid = move.avatar_id()
    if aid is None:
        return None
    objs = {o.id: o for o in scene.objects}
    avatar = objs.get(aid)
    if avatar is None:
        return None
    skip = reached_ids or set()
    others = [o for o in scene.objects if o.id != aid and o.id not in skip]
    if not others:
        return None
    freq = Counter(o.color for o in scene.objects)
    ay, ax = avatar.centroid
    target = _pick_target(others, avatar, freq, target_color)
    ty, tx = target.centroid
    cur_dist = abs(ty - ay) + abs(tx - ax)
    # Alvo alcancado (distancia < 3px): marca como reached e tenta o proximo
    if cur_dist < 3 and reached_ids is not None:
        reached_ids.add(target.id)
        remaining = [o for o in others if o.id != target.id]
        if not remaining:
            return None
        target = _pick_target(remaining, avatar, freq, target_color)
        ty, tx = target.centroid
        cur_dist = abs(ty - ay) + abs(tx - ax)
    best, bd = None, cur_dist
    blocked = getattr(move, "blocked", {})
    for k, (dr, dc) in moves.items():
        # Pula direcoes bloqueadas (bateu na parede 2+ vezes recentes)
        if blocked.get(k, 0) >= 2:
            continue
        nd = abs(ty - (ay + dr)) + abs(tx - (ax + dc))
        if nd < bd:
            bd, best = nd, k
    # Wall avoidance: se bloqueado na direcao desejada, tenta perpendicular.
    # Em vez de ficar batendo na parede, contorna. Nao faz nada se ja esta no alvo.
    if best is None and cur_dist > 2:
        dy, dx = ty - ay, tx - ax
        # Tenta direcao perpendicular ao eixo principal bloqueado
        perp_candidates = []
        for k, (dr, dc) in moves.items():
            if blocked.get(k, 0) >= 2:
                continue
            # Perpendicular: move no eixo que nao e o principal
            if abs(dy) >= abs(dx):
                # Precisa ir em Y mas esta bloqueado -> tenta X
                if dc != 0:
                    perp_candidates.append((k, abs(ty - (ay + dr)) + abs(tx - (ax + dc))))
            else:
                # Precisa ir em X mas esta bloqueado -> tenta Y
                if dr != 0:
                    perp_candidates.append((k, abs(ty - (ay + dr)) + abs(tx - (ax + dc))))
        if perp_candidates:
            best = min(perp_candidates, key=lambda x: x[1])[0]
    # Ultimo fallback: tenta qualquer direcao que aproxime
    if best is None:
        for k, (dr, dc) in moves.items():
            nd = abs(ty - (ay + dr)) + abs(tx - (ax + dc))
            if nd < cur_dist:
                bd, best = nd, k
                break
    return best
