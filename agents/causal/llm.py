# agents/causal/llm.py
from __future__ import annotations

import json
import os
import re
import threading
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
    'Pick the action from the Available list that best fits the scene; do not keep '
    'repeating an action that produced no visible change.'
)

_DIRECT_INSTRUCTION = (
    "Escolha a PROXIMA acao imediata (uma so) para fazer progresso. Responda "
    "APENAS um JSON, sem markdown, sem prosa:\n"
    '{"type":"press","action":"ACTIONk"}   (k da lista disponivel)\n'
    '{"type":"click_cell","gx":0,"gy":0}   (gx,gy em 0..5 = celula do grid 6x6)'
)


_HARMONY_TOKEN = re.compile(r"<\|[^|>]*\|>")
_HARMONY_FINAL = "<|channel|>final<|message|>"
_HARMONY_TERMS = ("<|return|>", "<|end|>", "<|start|>")


def _should_use_harmony(model_path) -> bool:
    """gpt-oss usa o formato Harmony (CoT no canal analysis, resposta no final);
    Qwen e demais nao. Detecta pelo slug do modelo."""
    return bool(model_path) and "gpt-oss" in str(model_path).lower()


def _strip_harmony_markers(s: str) -> str:
    return _HARMONY_TOKEN.sub("", s)


def _install_gpt_oss_kernel(kernel_dir) -> bool:
    """gpt-oss MXFP4 offline: entrega o pacote `triton_kernels` LOCAL direto ao
    transformers, monkeypatchando get_kernel. Contorna get_local_kernel, que exige
    um metadata.json que o repo nao publica (e list_repo_tree, que a rede bloqueia).
    kernel_dir = pasta que contem o pacote `triton_kernels/` (build/torch-universal).
    Devolve o MESMO objeto que get_local_kernel devolveria, sem tocar em rede/arquivo."""
    if not kernel_dir or not os.path.isdir(os.path.join(kernel_dir, "triton_kernels")):
        return False
    import importlib
    import sys
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    tk = importlib.import_module("triton_kernels")

    def _local_get_kernel(repo_id, *a, **k):
        if "triton_kernels" in str(repo_id):
            return tk
        raise RuntimeError("kernel nao-local pedido offline: " + str(repo_id))

    import transformers.integrations.hub_kernels as _hk
    _hk.get_kernel = _local_get_kernel
    try:  # mxfp4.py chama o nome get_kernel importado no seu proprio namespace
        import transformers.integrations.mxfp4 as _mx
        _mx.get_kernel = _local_get_kernel
    except Exception:
        pass
    return True


def extract_final_channel(text: str) -> str:
    """Isola o conteudo do canal `final` do Harmony (a resposta), descartando o
    canal `analysis` (a cadeia de raciocinio). Sem marcadores (modelo nao-harmony)
    -> devolve o texto limpo como veio. Robusto a alucinacao: o `parse_goal` a
    jusante ainda extrai o JSON."""
    i = text.rfind(_HARMONY_FINAL)
    if i == -1:
        return _strip_harmony_markers(text).strip()
    seg = text[i + len(_HARMONY_FINAL):]
    for term in _HARMONY_TERMS:
        j = seg.find(term)
        if j != -1:
            seg = seg[:j]
    return _strip_harmony_markers(seg).strip()


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

    def __init__(self, model_path, max_tokens: int = 512):
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

    def __init__(self, model_path, max_tokens: int = 512):
        import transformers.utils as _tu
        import transformers.utils.import_utils as _iu
        _tu.is_torchvision_available = lambda *a, **k: False   # pula torchvision quebrado
        _iu.is_torchvision_available = lambda *a, **k: False
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        self._harmony = _should_use_harmony(model_path)
        self._tok = AutoTokenizer.from_pretrained(model_path)
        if self._harmony:
            # gpt-oss ja vem em MXFP4 (~63GB, cabe). dtype="auto" preserva a quantizacao
            # (float16 forcaria dequant p/ bf16 ~234GB -> OOM). Entrega o pacote
            # triton_kernels LOCAL ao transformers (contorna get_kernel/rede/metadata.json).
            _ok = _install_gpt_oss_kernel(os.environ.get("GPT_OSS_KERNEL_DIR"))
            print("[causal] gpt-oss kernel local:", "OK" if _ok else "NAO INSTALADO")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype="auto", device_map="cuda")
        else:
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
        # gpt-oss RACIOCINA (canal analysis) antes do canal final -> precisa de orcamento
        # de tokens p/ ALCANCAR o final; Qwen roda com enable_thinking=False (curto).
        self._max = max(max_tokens, 2048) if self._harmony else max_tokens
        self._gen_lock = threading.Lock()   # serializa generate entre threads do Swarm

    def complete(self, prompt: str) -> str:
        import torch
        with self._gen_lock:                # model.generate não é thread-safe concorrente
            msgs = [{"role": "user", "content": prompt}]
            if self._harmony:
                # gpt-oss: aplica esforco de raciocinio; decodifica COM marcadores de
                # canal e extrai SO o canal `final` (a resposta), descartando o CoT.
                effort = os.environ.get("CAUSAL_EFFORT", "medium")
                try:
                    enc = self._tok.apply_chat_template(
                        msgs, add_generation_prompt=True, return_tensors="pt",
                        reasoning_effort=effort)
                except TypeError:
                    enc = self._tok.apply_chat_template(
                        msgs, add_generation_prompt=True, return_tensors="pt")
            else:
                try:  # Qwen3 pensa por padrão (<think>…</think>) e gasta os tokens; desliga.
                    enc = self._tok.apply_chat_template(
                        msgs, add_generation_prompt=True, return_tensors="pt",
                        enable_thinking=False)
                except TypeError:  # modelos sem esse kwarg (ex: Qwen2.5) ignoram e seguem
                    enc = self._tok.apply_chat_template(
                        msgs, add_generation_prompt=True, return_tensors="pt")
            ids = enc.input_ids if hasattr(enc, "input_ids") else enc   # 5.x: dict -> tensor
            ids = ids.to(self._model.device)
            attn = torch.ones_like(ids)                                 # evita warning pad==eos
            with torch.no_grad():
                out = self._model.generate(ids, attention_mask=attn,
                                           max_new_tokens=self._max, do_sample=False)
            if self._harmony:  # preserva marcadores p/ isolar o canal final
                text = self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
                return extract_final_channel(text)
            return self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def make_llm_client(model_path=None) -> LLMClient:
    """vLLM → transformers+4bit → NullLLMClient. Nunca levanta.
    gpt-oss (Harmony) pula o vLLM: o VLLMClient manda prompt cru sem chat template e
    o suporte Harmony do vLLM ainda e imaturo -> vai direto pro HFClient, que aplica
    o template Harmony + reasoning_effort e isola o canal final."""
    if model_path:
        if not _should_use_harmony(model_path):
            try:
                return VLLMClient(model_path)
            except Exception:
                pass
        try:
            return HFClient(model_path)
        except Exception:
            import traceback
            print("[causal] HFClient FALHOU ao carregar o modelo:")
            traceback.print_exc()
    return NullLLMClient()


_SHARED_LLM = None
_SHARED_KEY = "\0"
_SHARED_LLM_LOCK = threading.Lock()


def shared_llm_client(model_path=None) -> LLMClient:
    """Singleton por-processo: carrega o modelo UMA vez, compartilhado por todas as
    threads do Swarm. Sem isso, cada agente carregava seu próprio 7B → OOM → NullLLMClient."""
    global _SHARED_LLM, _SHARED_KEY
    with _SHARED_LLM_LOCK:
        if _SHARED_LLM is None or _SHARED_KEY != model_path:
            _SHARED_LLM = make_llm_client(model_path)
            _SHARED_KEY = model_path
        return _SHARED_LLM


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
    '  {"type":"code","source":"def decide(scene):\\n    return press(MOVES[0])"}'
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


_HEX = "0123456789abcdef"


def grid_to_ascii(grid) -> str:
    """Desenha o grid como ASCII: 1 char hex por celula (cor 0-15 -> 0-f), linhas
    separadas por \\n. E a representacao ESPACIAL que a lista de objetos nao da -
    o modelo ve a posicao exata de cada celula (approach do duck-v26/Tufa que pontuou).
    None/formato invalido -> string vazia."""
    if grid is None:
        return ""
    try:
        out = []
        for row in grid:
            out.append("".join(_HEX[int(v) & 15] for v in row))
        return "\n".join(out)
    except TypeError:
        return ""


def build_direct_prompt(scene, dyn, last=None) -> str:
    """Prompt orientado a ACAO (distinto de build_prompt, orientado a META): serializa
    o GRID ASCII (espacial) + a cena objeto-centrica + AVAILABLE_ACTIONS + feedback da
    ultima acao, e pede UMA proxima acao. O parsing reusa parse_goal; exec reusa execute_goal."""
    dyn = dyn or {}
    lines = ["GRID (1 char = cor 0-f; linha = y de cima->baixo, coluna = x):",
             grid_to_ascii(scene.grid),
             f"OBJETOS ({len(scene.objects)}):"]
    for o in scene.objects:
        lines.append(
            f"  id={o.id} color={o.color} centroid={o.centroid} "
            f"size={o.size} bbox={o.bbox}"
        )
    lines.append(f"AVAILABLE_ACTIONS: {dyn.get('available', [])}   (use SO essas)")
    if last and last.get("key"):
        eff = last.get("effect") or "nenhuma mudanca"
        lines.append(
            f"Sua ultima acao {last['key']} produziu: {eff}. Escolha a PROXIMA "
            "acao que faz PROGRESSO; NAO repita uma acao que nao mudou nada."
        )
    lines.append(_DIRECT_INSTRUCTION)
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
