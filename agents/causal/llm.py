# agents/causal/llm.py
from __future__ import annotations

import json
from collections import Counter

GOAL_TYPES = {"press", "click_cell", "reach", "code"}

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

    def complete_many(self, prompt: str, n: int) -> list:
        # amostragem massiva: impl padrão = n chamadas. VLLMClient/HFClient podem
        # sobrescrever com n-amostras reais (temperatura>0) numa só chamada.
        return [self.complete(prompt) for _ in range(n)]


class NullLLMClient(LLMClient):
    def complete(self, prompt: str) -> str:
        return ""


class VLLMClient(LLMClient):
    """Serving via vLLM (primário). Import lazy — llm.py fica offline-safe."""

    def __init__(self, model_path, max_tokens: int = 256):
        from vllm import LLM, SamplingParams  # lazy
        self._llm = LLM(model=model_path, dtype="float16",
                        gpu_memory_utilization=0.9)
        self._sp = SamplingParams(temperature=0.2, max_tokens=max_tokens)

    def complete(self, prompt: str) -> str:
        out = self._llm.generate([prompt], self._sp)
        return out[0].outputs[0].text


class HFClient(LLMClient):
    """Fallback: transformers + bitsandbytes (4-bit). Import lazy."""

    def __init__(self, model_path, max_tokens: int = 256):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)  # lazy
        self._tok = AutoTokenizer.from_pretrained(model_path)
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto")
        self._max = max_tokens

    def complete(self, prompt: str) -> str:
        import torch
        ids = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(ids, max_new_tokens=self._max,
                                       do_sample=False)
        return self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def make_llm_client(model_path=None) -> LLMClient:
    """vLLM → transformers+4bit → NullLLMClient. Nunca levanta."""
    if model_path:
        try:
            return VLLMClient(model_path)
        except Exception:
            pass
        try:
            return HFClient(model_path)
        except Exception:
            pass
    return NullLLMClient()


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
    if t == "code" and isinstance(g.get("source"), str):
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
    if t == "code":
        from .sandbox import execute_code_goal
        return execute_code_goal(goal.get("source", ""), scene)
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
