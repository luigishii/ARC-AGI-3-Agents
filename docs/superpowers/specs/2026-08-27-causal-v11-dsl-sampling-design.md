# CausalObjectAgent v11 — DSL numpy + Amostragem Massiva & Ranking · Design

> Roadmap pós-v10: (a) uma **DSL de primitivas de abstração visual** injetada no
> namespace do sandbox, e (b) **amostragem de múltiplos candidatos** do LLM +
> **ranking rápido** contra o nosso forward-model. Fecha o item v11.

## Discrepância factual (importante) e o análogo fiel

ARC-AGI-3 é **interativo**: NÃO há "pares de treino" input→output (isso é
AGI-1/2). Não dá pra rankear candidatos contra demonstrações rotuladas. O
**análogo fiel** — e que roda em **ms** — é rankear cada `decide(scene)`
candidato **contra a nossa experiência acumulada**: simular a 1ª ação do
candidato pelo `CausalModel`/`TransitionModel` (lookups em dict) e pontuar por
novidade/controlabilidade/proximidade-de-âncora. **Suposição a confirmar na
revisão.**

## Restrição do ambiente

Sem GPU/LLM aqui → TDD com `FakeLLM` (que devolve candidatos canned) e código
canned. `dsl.py`/ranker são stdlib+numpy, 100% testáveis localmente.

## Decomposição

- **v11a — DSL (`dsl.py`) + injeção no sandbox.**
- **v11b — amostragem (`LLMClient.complete_many`) + `rank_candidates`.**
- (v11d, wiring no controlador — sub-projeto posterior, fora deste spec.)

## v11a — DSL de abstração visual

Novo `agents/causal/dsl.py`: funções puras sobre `scene` (que tem `.objects` e
`.grid`) + construtores de ação. Um dict `DSL = {nome: função}` é injetado no
namespace do sandbox, de modo que o `decide(scene)` do LLM possa compor
primitivas testadas em vez de reinventar.

Primitivas (mínimas, generalizáveis):
- **Consultas de objeto:** `objects_of_color(scene, c)`, `rarest_color(scene)`,
  `largest(objs)`, `smallest(objs)`, `nearest(objs, point)`.
- **Acessores:** `ocolor(o)`, `osize(o)`, `oid(o)`, `ocentroid(o)` → `(r,c)`.
- **Espaciais:** `manhattan(a, b)` (pontos), `same_color(a, b)`.
- **Construtores de ação (retornam `action_key`):** `press(name)` → `name`;
  `click(gx, gy)` → `"ACTION6@cell=gx,gy"`; `move_toward(avatar, target)` →
  usa o `MOVES` injetado (dict `ação→(dr,dc)`) e devolve a ação que aproxima
  (mesma lógica gulosa do navigate), ou `None`.

### Injeção no sandbox (`sandbox.py`)

`compile_decide(source, extra=None)` e `run_decide(fn, scene, timeout, ...)`
passam a aceitar `extra` (dict) fundido no namespace de exec (`g.update(extra)`).
`execute_code_goal(source, scene, timeout, extra=None)` idem. O agente, ao rodar
uma meta `code`, injeta `{**DSL, "MOVES": self._move.moves()}`. Sem `extra`,
comportamento idêntico ao v10c (retrocompatível).

## v11b — Amostragem massiva + ranking

### Amostragem (`llm.py`)

`LLMClient.complete_many(prompt, n) -> list[str]`: interface com **impl padrão**
que chama `complete()` `n` vezes (funciona p/ qualquer cliente; `NullLLMClient`
→ lista de `""`). `VLLMClient`/`HFClient` sobrescrevem com amostragem real
(temperatura>0, `n` saídas numa chamada — vLLM faz isso nativo). `FakeLLM` de
teste devolve uma lista canned.

### Ranking (`ranker.py`)

`rank_candidates(sources, scene, model, tmodel, novelty, moves, extra) -> str|None`:
- para cada `source`: `execute_code_goal(source, scene, extra=...)` → a 1ª
  `action_key` (compila+roda em ms, com timeout; inválido → descarta o
  candidato);
- **score simulado** dessa ação (sem gastar ação real):
  `eff, conf = model.predict(key)`; `nxt = tmodel.predict_next(sig, key)`;
  `score = novelty.novelty(nxt if nxt else sig) * ctrl` (ctrl=conf; par inédito
  → score otimista) — reusa a lógica de `plan`/`score`;
- retorna o `source` de **maior score** (empate → 1º; nenhum válido → `None`).

Tudo são lookups em dict → **ms** para dezenas de candidatos.

## Fluxo de dados (visão v11d)

Ao consultar o LLM p/ uma meta `code`: `complete_many(prompt, N)` → N sources →
`rank_candidates(...)` escolhe o melhor pelo forward-model → vira `self._goal`
(`{"type":"code","source":<melhor>}`) → executado por-passo no sandbox com a DSL
injetada → validação/re-pergunta do controlador (v10d).

## Erros e casos de borda

- **`extra=None`:** sandbox idêntico ao v10c.
- **Candidato inválido/timeout:** descartado no ranking (não quebra os demais).
- **Nenhum candidato válido:** `rank_candidates` → `None` → fallback.
- **`complete_many` sem modelo (`NullLLMClient`):** `[""]*n` → todos parseiam p/
  None → `None` → fallback determinístico.
- **Determinismo:** dado sources+scene+modelos, o ranking é determinístico.

## TDD (plano)

**Task 1 — DSL (`test_dsl.py`):** cada primitiva (objects_of_color, rarest_color,
nearest, ocentroid, manhattan, press, click, move_toward com MOVES) com cenas de
brinquedo.

**Task 2 — injeção no sandbox (`test_sandbox_dsl.py`):** `compile_decide`/
`execute_code_goal` com `extra` fundem a DSL no namespace; um `decide` que chama
`click(2,3)` → `"ACTION6@cell=2,3"`; sem `extra` = comportamento v10c.

**Task 3 — amostragem (`test_llm_sampling.py`):** `LLMClient.complete_many`
default chama `complete` n vezes; `NullLLMClient.complete_many` → `[""]*n`;
`FakeLLM` de teste devolve lista.

**Task 4 — ranking (`test_ranker.py`):** `rank_candidates` escolhe o source cuja
1ª ação tem maior score simulado; descarta inválidos; nenhum válido → `None`.

**Regressão:** os 170 testes v1–v10c seguem verdes em cada commit.

## Fora de escopo

- **v11d:** wiring do sampling+ranking no controlador (`agent.py`) + notebook
  shipa `dsl.py`/`ranker.py`.
- **v12:** MCTS no espaço de código + autorreparo.
- DSL mais rica (transformações de grade, simetrias) — iterar depois.

## Critério de pronto (v11a+b)

- `dsl.py` com as primitivas; sandbox injeta `extra`; `complete_many`;
  `rank_candidates` pontua via forward-model em ms.
- 170 testes v1–v10c + novos verdes.
- **Pendente de confirmação:** o ranking usa o forward-model aprendido (não
  "pares de treino", que AGI-3 não tem).
