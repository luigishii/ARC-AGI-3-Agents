# agents/causal/llm.py
from __future__ import annotations

import json
from collections import Counter

GOAL_TYPES = {"press", "click_cell", "reach"}

_INSTRUCTION = (
    "You are playing a grid puzzle. Infer the GOAL and reply with ONLY a JSON "
    "object, one of:\n"
    '{"type":"press","action":"ACTION1"}\n'
    '{"type":"click_cell","gx":0,"gy":0}\n'
    '{"type":"reach","avatar":<sel>,"target":<sel>}  '
    '(sel = {"id":I} | {"color":C} | "rarest")'
)


class LLMClient:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class NullLLMClient(LLMClient):
    def complete(self, prompt: str) -> str:
        return ""


def build_prompt(scene, dynamics) -> str:
    dyn = dynamics or {}
    lines = [f"OBJECTS ({len(scene.objects)}):"]
    for o in scene.objects:
        lines.append(
            f"  id={o.id} color={o.color} centroid={o.centroid} "
            f"size={o.size} bbox={o.bbox}"
        )
    lines.append(f"AVAILABLE_ACTIONS: {dyn.get('available', [])}")
    lines.append(f"DYNAMICS: moves={dyn.get('moves', {})} notes={dyn.get('notes', '')}")
    lines.append(_INSTRUCTION)
    return "\n".join(lines)


def parse_goal(text):
    if not text:
        return None
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        g = json.loads(text[i:j + 1])
    except Exception:
        return None
    if not isinstance(g, dict) or g.get("type") not in GOAL_TYPES:
        return None
    t = g["type"]
    if t == "press" and "action" in g:
        return g
    if t == "click_cell" and "gx" in g and "gy" in g:
        return g
    if t == "reach" and "avatar" in g and "target" in g:
        return g
    return None


def _resolve(sel, objects):
    if not objects:
        return None
    if sel == "rarest":
        freq = Counter(o.color for o in objects)
        return min(objects, key=lambda o: freq[o.color])
    if isinstance(sel, dict):
        if "id" in sel:
            for o in objects:
                if o.id == sel["id"]:
                    return o
            return None
        if "color" in sel:
            ms = [o for o in objects if o.color == sel["color"]]
            return ms[0] if ms else None
    return None


def execute_goal(goal, scene, moves):
    t = goal.get("type")
    if t == "press":
        return goal.get("action")
    if t == "click_cell":
        return f"ACTION6@cell={goal['gx']},{goal['gy']}"
    if t == "reach":
        if not moves:
            return None
        avatar = _resolve(goal.get("avatar"), scene.objects)
        if avatar is None:
            return None
        others = [o for o in scene.objects if o.id != avatar.id]
        target = _resolve(goal.get("target"), others)
        if target is None:
            return None
        ay, ax = avatar.centroid
        ty, tx = target.centroid
        best, bd = None, abs(ty - ay) + abs(tx - ax)
        for k, (dr, dc) in moves.items():
            nd = abs(ty - (ay + dr)) + abs(tx - (ax + dc))
            if nd < bd:
                bd, best = nd, k
        return best
    return None
