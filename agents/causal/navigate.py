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

    def observe(self, key, prev, curr) -> None:
        m = _moved_object(prev, curr)
        if m is None:
            return
        oid, v = m
        self.vec.setdefault(key, {})
        self.vec[key][v] = self.vec[key].get(v, 0) + 1
        self.avatar_counts[oid] = self.avatar_counts.get(oid, 0) + 1

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


def navigate(scene, move):
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
    others = [o for o in scene.objects if o.id != aid]
    if not others:
        return None
    freq = Counter(o.color for o in scene.objects)
    ay, ax = avatar.centroid
    target = min(others, key=lambda o: (freq[o.color],
                                        abs(o.centroid[0] - ay) + abs(o.centroid[1] - ax)))
    ty, tx = target.centroid
    best, bd = None, abs(ty - ay) + abs(tx - ax)
    for k, (dr, dc) in moves.items():
        nd = abs(ty - (ay + dr)) + abs(tx - (ax + dc))
        if nd < bd:
            bd, best = nd, k
    return best
