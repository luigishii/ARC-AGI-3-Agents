# CausalObjectAgent v10b — Serving concreto do LLM + empacotamento · Design

> Passo v10b. O v10a entregou o contrato (`build_prompt`/`parse_goal`/
> `execute_goal`, `LLMClient`/`NullLLMClient`). Agora: **cliente concreto** que
> serve o Qwen2.5-Coder-7B-Instruct offline no Kaggle, e o **empacotamento** no
> notebook. Decisões do usuário: vLLM primário, transformers+4-bit fallback.

## Restrição honesta

Sem GPU/LLM/internet aqui → o TDD cobre só o **factory defensivo** e o
**empacotamento** (o `llm.py` continuar offline-safe, o notebook embutir o
módulo e as envs). O carregamento/geração real do modelo só roda no Kaggle. Os
pesos entram como **Kaggle Dataset que o usuário sobe** (ação manual). O agente
**só invoca o LLM no v10d** (wiring) — v10b prepara serving + pacote.

## Arquitetura

Mudanças em `agents/causal/llm.py` (clientes concretos + factory, **lazy
import**) e `kaggle/build_notebook.py` (empacota `llm.py` + envs). Nenhum outro
módulo muda. `llm.py` mantém o **topo stdlib-only** (offline-safe); `vllm`/
`torch`/`transformers` só são importados **dentro** dos `__init__` dos clientes.

### 1. Clientes concretos (`llm.py`)

```python
class VLLMClient(LLMClient):
    def __init__(self, model_path, max_tokens=256):
        from vllm import LLM, SamplingParams            # lazy
        self._llm = LLM(model=model_path, dtype="float16",
                        gpu_memory_utilization=0.9)
        self._sp = SamplingParams(temperature=0.2, max_tokens=max_tokens)
    def complete(self, prompt):
        out = self._llm.generate([prompt], self._sp)
        return out[0].outputs[0].text


class HFClient(LLMClient):
    def __init__(self, model_path, max_tokens=256):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)   # lazy
        self._tok = AutoTokenizer.from_pretrained(model_path)
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.float16)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto")
        self._max = max_tokens
    def complete(self, prompt):
        import torch
        ids = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(ids, max_new_tokens=self._max,
                                       do_sample=False)
        return self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def make_llm_client(model_path=None):
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
```

`make_llm_client` **nunca levanta** — qualquer falha (libs ausentes, OOM,
caminho inválido) cai pro `NullLLMClient` (agente roda determinístico). Nesta
máquina (sem `vllm`/`transformers`-de-geração) o factory retorna `NullLLMClient`
— **testável**.

### 2. Empacotamento no notebook (`build_notebook.py`)

- Adicionar `"llm.py"` à lista `MODULES` (o `.py` já está em `agents/causal/`) →
  embutido no notebook.
- No `.env` do rerun, acrescentar:
  - `CAUSAL_LLM=1` (liga o híbrido — inerte até o v10d).
  - `QWEN_MODEL_PATH=<caminho do dataset de pesos>` (constante `MODEL_DATASET_PATH`
    no build-script; **o usuário edita** pro slug do seu dataset e regenera o
    notebook).
- **Não** anexa o dataset automaticamente (o usuário anexa o dataset de pesos ao
  notebook pela UI do Kaggle) — o notebook só referencia o caminho via env.

## Erros e casos de borda

- **Libs de serving ausentes / OOM / caminho inválido:** `make_llm_client` →
  `NullLLMClient` → fallback determinístico. Sem crash.
- **`llm.py` offline-safe:** topo importa só stdlib; `import agents.causal.llm`
  funciona sem `vllm`/`torch` (testado).
- **Notebook sem o dataset anexado:** `QWEN_MODEL_PATH` aponta p/ caminho
  inexistente → `make_llm_client` cai pro `NullLLMClient` → roda determinístico.

## Testes (TDD)

1. **`import agents.causal.llm` não puxa `vllm`/`torch`** (`test_llm_serving.py`):
   após importar o módulo, `sys.modules` não contém `vllm` nem `torch`.
2. **`make_llm_client(None)` → `NullLLMClient`.**
3. **`make_llm_client("/nao/existe")` → `NullLLMClient`** (VLLMClient/HFClient
   falham no import/load → capturado).
4. **`VLLMClient`/`HFClient` existem** e são subclasses de `LLMClient`.
5. **notebook** (`test_build_notebook.py`, estender): as `MODULES` incluem
   `"llm.py"`; o notebook gerado contém `agents/causal/llm.py`, `CAUSAL_LLM` e
   `QWEN_MODEL_PATH`.
6. **Regressão:** os 138 testes v1–v10a seguem verdes.

## Fora de escopo

- **v10d:** wiring do LLM no loop de decisão do agente (controlador: quando
  chamar, validar hipótese, re-perguntar). Sem isso o LLM não é invocado no run.
- Upload/anexo do dataset de pesos (ação manual do usuário).
- Medir latência/caber em 9h (validação no Kaggle).

## Critério de pronto

- `VLLMClient`/`HFClient`/`make_llm_client` com lazy import; `llm.py`
  offline-safe; notebook empacota `llm.py` + envs.
- 138 testes v1–v10a + novos verdes.
- **Runbook do usuário** documentado: subir pesos como Dataset, anexar ao
  notebook, editar `MODEL_DATASET_PATH`, regenerar o notebook. LLM só ativo
  após v10d.
