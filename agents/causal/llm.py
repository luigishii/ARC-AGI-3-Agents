# agents/causal/llm.py
from __future__ import annotations

import json
from collections import Counter

GOAL_TYPES = {"press", "click_cell", "reach", "code"}

_INSTRUCTION = (
    "You are playing a grid puzzle. Infer the GOAL and reply with ONLY a JSON "
    "object — no markdown fences, no prose, no explanation — one of:\n"
    '{"type":"press","action":"ACTION1"}\n'
    '{"type":"click_cell","gx":0,"gy":0}\n'
    '{"type":"reach","avatar":<sel>,"target":<sel>}  '
    '(sel = {"id":I} | {"color":C} | "rarest")\n'
    '{"type":"code","source":"def decide(scene):\\n    return \'ACTION1\'"}\n'
    'Example valid reply: {"type":"press","action":"ACTION2"}'
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
    """Fallback: transformers. Tenta bitsandbytes 4-bit; se ausente, carrega FP16.
    Desliga torchvision (o image_utils do transformers 5.x quebra com o PIL/torchvision
    da imagem Kaggle: 'cannot import name _Ink'). Import lazy → llm.py fica offline-safe."""

    def __init__(self, model_path, max_tokens: int = 256):
        import transformers.utils as _tu
        import transformers.utils.import_utils as _iu
        _tu.is_torchvision_available = lambda *a, **k: False   # pula torchvision quebrado
        _iu.is_torchvision_available = lambda *a, **k: False
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        self._tok = AutoTokenizer.from_pretrained(model_path)
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401 — só usa 4-bit se a lib existir
            bnb = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_compute_dtype=torch.float16)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb, device_map="auto")
        except Exception:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.float16, device_map="auto")   # transformers 5.x: dtype
        self._max = max_tokens

    def complete(self, prompt: str) -> str:
        import torch
        enc = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt")
        ids = enc.input_ids if hasattr(enc, "input_ids") else enc   # 5.x: dict -> tensor
        ids = ids.to(self._model.device)
        attn = torch.ones_like(ids)                                 # evita warning pad==eos
        with torch.no_grad():
            out = self._model.generate(ids, attention_mask=attn,
                                       max_new_tokens=self._max, do_sample=False)
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


def client_kind(client) -> str:
    """Rótulo do cliente ativo p/ diagnóstico ('null' = degradou → LLM não subiu)."""
    return {"NullLLMClient": "null", "VLLMClient": "vllm", "HFClient": "hf"}.get(
        type(client).__name__, type(client).__name__.lower())


_FEWSHOT = (
    "EXAMPLES (code goals use DSL helpers: rarest_color/objects_of_color/largest/"
    "nearest/click/press/move_toward):\n"
    '  {"type":"code","source":"def decide(scene):\\n    c = rarest_color(scene)\\n'
    '    o = largest(objects_of_color(scene, c))\\n    r, k = ocentroid(o)\\n'
    '    return click(int(k)//11, int(r)//11)"}\n'
    '  {"type":"code","source":"def decide(scene):\\n    return press(\'ACTION2\')"}'
)


def build_prompt(scene, dynamics) -> str:
    dyn = dynamics or {}
    lines = [f"OBJECTS ({len(scene.objects)}):"]
    for o in scene.objects:
        lines.append(
            f"  id={o.id} color={o.color} centroid={o.centroid} "
            f"size={o.size} bbox={o.bbox}"
        )
    lines.append(f"AVAILABLE_ACTIONS: {dyn.get('available', [])}   (use ONLY these)")
    lines.append(f"DYNAMICS: moves={dyn.get('moves', {})} notes={dyn.get('notes', '')}")
    lines.append(_FEWSHOT)
    lines.append(_INSTRUCTION)
    return "\n".join(lines)


def _extract_json(text):
    """Extrai o 1º objeto JSON de uma resposta de LLM, tolerando cercas markdown
    (```json ... ```) e prosa em volta. Balanceia chaves; cai p/ 1º{...último} se falhar."""
    if not text:
        return None
    s = text.strip()
    if "```" in s:                                   # tira cercas markdown
        blocks = [b for b in s.split("```") if "{" in b]
        if blocks:
            s = max(blocks, key=len)
            if s.lstrip()[:4].lower() == "json":
                s = s.lstrip()[4:]
    i = s.find("{")
    if i < 0:
        return None
    depth = 0                                        # varre até fechar o 1º objeto
    for k in range(i, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[i:k + 1])
                except Exception:
                    break
    j = s.rfind("}")                                 # fallback: 1º { ... último }
    if j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None


def parse_goal(text):
    g = _extract_json(text)
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
        from .dsl import DSL
        return execute_code_goal(goal.get("source", ""), scene,
                                 extra={**DSL, "MOVES": moves})
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
