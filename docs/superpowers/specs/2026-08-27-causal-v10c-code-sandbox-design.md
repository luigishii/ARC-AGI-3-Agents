# CausalObjectAgent v10c — Sandbox de código Python do LLM · Design

> Estende o híbrido (v10a–d): além de metas declarativas, o LLM pode devolver
> uma **função `decide(scene) -> action_key`** que rodamos num sandbox e usamos
> como política por-passo. Fecha o item "código/rule em sandbox" do roadmap.

## Threat model (honesto)

O código vem do **nosso próprio LLM** (Qwen que controlamos), resolvendo puzzles
— não é um adversário. O sandbox protege contra **acidente/alucinação/erro/
loop infinito**, capturando tudo e caindo pro fallback. **NÃO** é uma barreira
de segurança à prova de código hostil (o `exec` do Python tem escapes via
dunder). Endurecer com isolamento de processo é follow-up, se necessário.

## Restrição do ambiente

Sem GPU/LLM aqui → TDD com código canned (não precisa de modelo). O `sandbox.py`
é stdlib puro, 100% testável localmente.

## Escopo

1. `agents/causal/sandbox.py`: compila e roda `decide(scene)` num namespace
   restrito, com timeout e captura de exceções → `None` no fallback.
2. `llm.py`: novo tipo de meta `code` (`parse_goal` valida; `execute_goal` roda
   via sandbox). **`agent.py` NÃO muda** (o controlador já chama `execute_goal`
   por-passo — uma meta `code` vira uma política persistente até invalidar).

## Arquitetura

### 1. `sandbox.py`

```python
SAFE_BUILTINS = {  # whitelist — sem import/open/eval/exec/file
    "len","min","max","sorted","range","abs","sum","enumerate","list","dict",
    "set","tuple","int","float","str","bool","any","all","map","filter","zip",
    "round","print"  # print vira no-op
}  # (mapeado p/ as funções reais; print = lambda *a,**k: None)

def compile_decide(source) -> callable | None:
    # exige "decide" no source; exec num namespace {"__builtins__": SAFE_BUILTINS};
    # retorna a função decide, ou None se compilar/definir falhar (try/except).

def run_decide(fn, scene, timeout=0.5) -> str | None:
    # roda fn(scene) numa daemon-thread com join(timeout); captura QUALQUER
    # exceção; se estourar o tempo (hang) → None (abandona a thread); só retorna
    # str (action_key), senão None.

def execute_code_goal(source, scene, timeout=0.5) -> str | None:
    # compile_decide + run_decide. Ponto de entrada usado pelo execute_goal.
```

- **Import bloqueado:** sem `__import__` nos builtins → `import` levanta
  `ImportError` → capturado → `None`.
- **Timeout via thread:** `signal.alarm` não serve (o `Swarm` roda em threads
  não-main). Usamos daemon-thread + `join(timeout)`; um `while True` é abandonado
  (não bloqueia o run). **Tradeoff honesto:** a thread abandonada continua
  rodando (vaza CPU) até o fim do processo; mitigado por chamadas esparsas + a
  meta `code` ser invalidada rápido (contagem de falhas do controlador). Follow-up
  de endurecimento = pool de processos.
- **Retorno limpo:** só `str` (uma `action_key`) passa; qualquer outra coisa,
  exceção, ou timeout → `None` → o controlador conta falha e cai no fallback.

### 2. `llm.py` — tipo de meta `code`

- `GOAL_TYPES` ganha `"code"`.
- `parse_goal`: `{"type":"code","source":"..."}` válido exige `"source"` (str).
- `execute_goal`: `if t == "code": from .sandbox import execute_code_goal;
  return execute_code_goal(goal["source"], scene)`. (Import local pra não puxar
  nada pesado — `sandbox` é stdlib, mas mantém `llm.py` enxuto.)

O código do LLM recebe `scene` (a cena objeto-cêntrica já parseada, com
`.objects`) e deve retornar uma **`action_key`** (ex.: `"ACTION1"` ou
`"ACTION6@cell=2,3"`), que o agente mapeia pro `Candidate` como qualquer meta.

## Fluxo de dados

LLM → `{"type":"code","source":"def decide(scene): ..."}` → `parse_goal` valida →
vira `self._goal` (persistente) → a cada passo `execute_goal` → `sandbox`
compila+roda `decide(scene)` com timeout → `action_key|None`. `None` → o
controlador conta falha, invalida após `GOAL_FAIL_MAX` e re-pergunta.

## Erros e casos de borda

- **Sintaxe inválida / sem `decide`:** `compile_decide` → `None`.
- **Exceção em runtime / import / acesso proibido:** capturado → `None`.
- **Loop infinito / hang:** timeout → `None` (thread abandonada).
- **Retorno não-string** (None, número, ação inválida): `None` → fallback (o
  agente valida `gkey in keymap`, então uma string inválida também cai fora).
- **Determinismo:** dado o mesmo `source` e `scene`, o resultado é determinístico
  (sem `random` nos builtins).

## Testes (TDD, `tests/causal/test_sandbox.py` + estender `test_llm*`)

1. **`compile_decide`** válido → callable; inválido (syntax) → `None`; sem
   `decide` → `None`.
2. **`run_decide`** roda e retorna a `action_key`; exceção → `None`; retorno
   não-str → `None`.
3. **import bloqueado:** `source` com `import os` → `None`.
4. **timeout:** `decide` com `while True: pass` + `timeout=0.1` → `None`
   (não trava o teste).
5. **`execute_code_goal`** end-to-end (source → action_key).
6. **`parse_goal`** aceita `{"type":"code","source":"..."}`; rejeita sem
   `source`.
7. **`execute_goal`** tipo `code` → roda o `decide` e devolve a `action_key`.
8. **Regressão:** os 158 testes v1–v10d seguem verdes.

## Fora de escopo

- Isolamento de processo / kill de threads (endurecimento).
- Amostragem massiva de código (v11) e MCTS/autorreparo (v12).

## Critério de pronto

- `sandbox.py` compila/roda `decide` com timeout e captura total → `None`
  limpo; `llm.py` com meta `code`; agente inalterado (usa `execute_goal`).
- 158 testes v1–v10d + novos verdes.
