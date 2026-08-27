# agents/causal/causal_model.py
from __future__ import annotations

from collections import namedtuple

Effect = namedtuple("Effect", "kind detail")


def compute_effect(prev, curr) -> Effect:
    prev_by_id = {o.id: o for o in prev.objects}
    curr_by_id = {o.id: o for o in curr.objects}
    changes = []
    for oid, po in prev_by_id.items():
        co = curr_by_id.get(oid)
        if co is None:
            changes.append(Effect("disappeared", oid))
        else:
            dr = round(co.centroid[0] - po.centroid[0])
            dc = round(co.centroid[1] - po.centroid[1])
            if (dr, dc) != (0, 0):
                changes.append(Effect("moved", (dr, dc)))
            elif co.color != po.color:
                changes.append(Effect("recolored", (po.color, co.color)))
    for oid, co in curr_by_id.items():
        if oid not in prev_by_id:
            changes.append(Effect("appeared", oid))
    if not changes:
        return Effect("none", None)
    if len(changes) == 1:
        return changes[0]
    return Effect("structural", tuple(c.kind for c in changes))
