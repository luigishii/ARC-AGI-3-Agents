# agents/causal/dsl.py
# Primitivas de abstração visual injetadas no namespace do sandbox, p/ o
# decide(scene) do LLM compor blocos semânticos testados.
from __future__ import annotations

from collections import Counter


def objects_of_color(scene, c):
    return [o for o in scene.objects if o.color == c]


def rarest_color(scene):
    if not scene.objects:
        return None
    freq = Counter(o.color for o in scene.objects)
    return min(freq, key=lambda k: freq[k])


def largest(objs):
    return max(objs, key=lambda o: o.size) if objs else None


def smallest(objs):
    return min(objs, key=lambda o: o.size) if objs else None


def nearest(objs, point):
    if not objs:
        return None
    pr, pc = point
    return min(objs, key=lambda o: abs(o.centroid[0] - pr) + abs(o.centroid[1] - pc))


def ocolor(o):
    return o.color


def osize(o):
    return o.size


def oid(o):
    return o.id


def ocentroid(o):
    return o.centroid


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def same_color(a, b):
    return a.color == b.color


def press(name):
    return name


def click(gx, gy):
    return f"ACTION6@cell={gx},{gy}"


def move_toward(avatar, target, moves):
    if avatar is None or target is None or not moves:
        return None
    ay, ax = avatar.centroid
    ty, tx = target.centroid
    best, bd = None, abs(ty - ay) + abs(tx - ax)
    for k, (dr, dc) in moves.items():
        nd = abs(ty - (ay + dr)) + abs(tx - (ax + dc))
        if nd < bd:
            bd, best = nd, k
    return best


DSL = {
    "objects_of_color": objects_of_color, "rarest_color": rarest_color,
    "largest": largest, "smallest": smallest, "nearest": nearest,
    "ocolor": ocolor, "osize": osize, "oid": oid, "ocentroid": ocentroid,
    "manhattan": manhattan, "same_color": same_color,
    "press": press, "click": click, "move_toward": move_toward,
}
