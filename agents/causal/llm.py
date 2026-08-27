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
