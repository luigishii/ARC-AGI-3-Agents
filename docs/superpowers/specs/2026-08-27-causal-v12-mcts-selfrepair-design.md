# CausalObjectAgent v12 — Busca em Espaço de Código (MCTS) + Autorreparo · Design

> Passo final do roadmap híbrido. v11 gera hipóteses one-shot; v12 traz **busca
> profunda** (expandir nós promissores) e **autorreparo** (test-time debugging:
> reenviar o erro/mismatch ao LLM p/ corrigir o código). **Apenas spec + plano
> TDD** — execução pendente (recomendação: validar v10-v11 no Kaggle antes).

## Restrição honesta (repetida)

Sem GPU/LLM aqui → TDD com `FakeLLM`/mocks. A busca e o reparo são testáveis no
encanamento; a **qualidade do raciocínio real só no Kaggle**. E: v10-v11 ainda
**nunca rodaram com modelo real** — o v12 empilha sobre camadas não-validadas.

## Enquadramento honesto do "MCTS"

MCTS clássico expande uma árvore de estados do jogo. Em AGI-3 não podemos
simular o jogo de graça — só temos o **forward-model aprendido**
(`TransitionModel`). Então v12 = **busca best-first no espaço de CÓDIGO**,
guiada por **rollouts no forward-model** e **expansão dirigida pelo LLM** (refino
dos nós promissores). É a variante fiel de MCTS pra AGI-3. Custo: chamadas ao
LLM na expansão são caras → orçamento **por-nível, não por-passo**.

## Decomposição

- **v12a — rollout + busca (`search.py`).**
- **v12b — autorreparo (`repair.py`).**
- **v12d — wiring no controlador** (sob envs `CAUSAL_MCTS`/`CAUSAL_REPAIR`).

## v12a — Rollout no forward-model + busca best-first

Novo `agents/causal/search.py`.

### `rollout_score(source, scene, tmodel, novelty, moves, depth) -> float`

Estende o ranker do v11 de **1 passo** para **K passos simulados** (sem gastar
ação real):
- `sig = state_signature(scene)`; `total = 0`.
- repetir `depth` vezes: roda `execute_code_goal(source, scene, extra)` → `key`;
  se inválido → break; `nxt = tmodel.predict_next(sig, key)`;
  `total += novelty.novelty(nxt if nxt else sig)`; `sig = nxt or sig`.
  (Como a cena não é simulável objeto-a-objeto, usamos a **assinatura** como
  estado; o `decide` recebe a cena real do passo 0 — aproximação honesta: o
  rollout mede quão "produtiva/nova" a política é na dinâmica aprendida.)
- retorna `total` (score cumulativo).

### `mcts_search(roots, refine_fn, scene, tmodel, novelty, moves, budget, depth) -> str|None`

- `pool = list(roots)`; avalia todos com `rollout_score`.
- `budget` iterações: pega o **melhor** do pool; `variants = refine_fn(best)`
  (o LLM refina/varia o nó promissor — `refine_fn` é injetado, mockável);
  avalia os variants e junta ao pool.
- retorna o `source` de maior score (nenhum válido → `None`).

`refine_fn(source) -> list[str]` é fornecido pelo controlador (encapsula o LLM);
nos testes é um mock que devolve variantes canned.

## v12b — Autorreparo (test-time debugging)

Novo `agents/causal/repair.py` (ou em `llm.py`).

### `build_repair_prompt(source, tried, observed, predicted, scene) -> str`

Monta o prompt de refino: o `source` atual, a ação tentada (`tried`), o que o
ambiente/forward-model respondeu (`observed`), o esperado (`predicted`), e a
cena — pedindo **só o JSON `{"type":"code","source":...}` corrigido**. Ex.:
"Sua função retornou X; o efeito foi Y (esperado Z); corrija o `decide`."

### Loop de reparo (no controlador, v12d)

Quando uma meta `code` (a) gera ação inválida (`execute_goal` None) **ou** (b) o
efeito observado ≠ previsto (mismatch detectado no fecha-loop via
`CausalModel.predict` vs `actual`):
- se `repair_count < REPAIR_MAX`: `resp = llm.complete(build_repair_prompt(...))`
  → `parse_goal` → se for `code` válido, substitui `self._goal`; `repair_count++`.
- senão: invalida a meta (fallback / re-pergunta normal).

Bounded (`REPAIR_MAX=2`) → não estoura o orçamento de chamadas.

## Erros e casos de borda

- **Forward-model esparso:** `rollout_score` com poucos dados → scores ~otimistas
  (fronteira); a busca ainda ordena por profundidade produtiva. Sem crash.
- **`refine_fn` devolve lixo:** variantes inválidas descartadas no `rollout_score`.
- **Reparo sem melhora:** limitado por `REPAIR_MAX` → cai no fallback determinístico.
- **Determinismo:** dado roots/mocks/modelos, busca e reparo são determinísticos.
- **Compat:** `CAUSAL_MCTS`/`CAUSAL_REPAIR` off por padrão → comportamento v11.

## TDD (plano)

**Task 1 — `rollout_score` (`test_search.py`):** política que alcança estados
novos em K passos pontua mais que uma saturada; código inválido → score baixo/
break.

**Task 2 — `mcts_search`:** com `refine_fn` mock que melhora um nó, a busca
retorna o melhor source após `budget` iterações; pool inicial ruim + refino bom
→ escolhe o refinado; nenhum válido → `None`.

**Task 3 — `build_repair_prompt` (`test_repair.py`):** o prompt contém o source,
a ação tentada, observado/esperado e pede JSON de código corrigido.

**Task 4 — loop de reparo no controlador (`test_agent_repair.py`, `FakeLLM`):**
meta `code` inválida + `CAUSAL_REPAIR=1` → dispara 1 reparo (novo source do
mock) e adota; após `REPAIR_MAX` sem sucesso → invalida; `CAUSAL_REPAIR=0` não
repara.

**Regressão:** os 186 testes v1–v11 verdes em cada commit.

## Fora de escopo

- Notebook shipar `search.py`/`repair.py` (parte do wiring quando executarmos).
- Simulação objeto-a-objeto da cena (usamos assinatura de estado — aproximação).

## Critério de pronto (quando executado)

- `rollout_score`/`mcts_search`/`build_repair_prompt` + loop de reparo, testados
  com mocks; envs `CAUSAL_MCTS`/`CAUSAL_REPAIR`/`CAUSAL_SAMPLES` compostos.
- 186 testes v1–v11 + novos verdes.
- **Recomendação:** só executar após validar v10-v11 com o LLM real no Kaggle.
