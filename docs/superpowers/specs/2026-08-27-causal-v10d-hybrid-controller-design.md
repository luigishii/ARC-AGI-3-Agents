# CausalObjectAgent v10d — Controlador Híbrido & Wiring do LLM · Design

> Fecha o pipeline híbrido: o `MovementModel`/`navigate` (do v9, aqui
> implementado) vira o executor de metas `reach`, e o **controlador** decide
> quando chamar o LLM (esparso), valida a hipótese e re-pergunta/faz fallback.
> Toda a fundação (v1–v8) e o contrato/serving (v10a/b) já existem.

## Restrição honesta

Sem GPU/LLM aqui → TDD com **`FakeLLM`** (respostas canned). Comportamento real
do modelo só no Kaggle (após o usuário anexar os pesos e ligar `CAUSAL_LLM`).

## Escopo

1. **`navigate.py`** (implementa o v9): `_moved_object`, `MovementModel`,
   `navigate` — reusados pelo `execute_goal` (`reach`) e como camada heurística.
2. **Controlador no `agent.py`**: chamadas esparsas ao LLM, meta ativa,
   validação (executável? nível subiu?), invalidação/re-pergunta, fallback.
3. **Pilha de decisão**: **LLM-goal → navigate → plan → greedy**.

## Arquitetura

Novo `agents/causal/navigate.py` + mudanças em `agents/causal/agent.py`.
`llm.py`/`planning.py`/`policy.py`/… não mudam (só consumidos).

### 1. `navigate.py` (v9)

`_moved_object(prev, curr) -> (id,(dr,dc))|None` (um único objeto casado que
transladou); `MovementModel` (`observe(key,prev,curr)`, `move_vector`, `moves()`
= só ações simples, `avatar_id()`, `to_dict/from_dict`); `navigate(scene, move)`
→ `action_key|None` (avatar = `avatar_id`; alvo = objeto não-avatar de cor mais
rara; ação gulosa que aproxima). Idêntico ao plano v9.

### 2. Controlador (`agent.py`)

Estado novo em `_init_causal_state`:
- `self._move = MovementModel()`, `self._nav_on = env CAUSAL_NAV != "0"`.
- `self._llm = make_llm_client(os.environ.get("QWEN_MODEL_PATH"))`.
- `self._llm_on = os.environ.get("CAUSAL_LLM", "0") != "0"` (default **off** local).
- `self._goal = None`, `self._goal_age = 0`, `self._goal_fails = 0`,
  `self._since_query = 10**9` (permite a 1ª consulta imediatamente).

Constantes: `QUERY_COOLDOWN = 8` (mín. de passos entre consultas ao LLM),
`GOAL_FAIL_MAX = 3` (metas inexecutáveis seguidas → invalida),
`GOAL_AGE_MAX = 20` (meta velha sem level-up → invalida).

Fecha-loop (após aprender transições): `self._move.observe(last_key, prev, scene)`;
**se o nível mudou** (`levels_completed` ≠ `_last_level`) → `self._goal = None`
(meta cumprida/novo nível → re-planejar).

Bloco de decisão (candidatos já montados; `keymap = {c.key: c}`):
```
self._since_query += 1
# (1) consulta esparsa: só se ligado, sem meta ativa e passado o cooldown
if self._llm_on and self._goal is None and self._since_query >= QUERY_COOLDOWN:
    dyn = {"available": [a.name for a in <available>],
           "moves": self._move.moves(), "notes": ""}
    self._goal = parse_goal(self._llm.complete(build_prompt(scene, dyn)))
    self._goal_age = 0
    self._since_query = 0
cand = None
# (2) executa a meta do LLM, com validação
if self._goal is not None:
    self._goal_age += 1
    gkey = execute_goal(self._goal, scene, self._move.moves())
    if gkey is not None and gkey in keymap:
        cand = keymap[gkey]; self._goal_fails = 0
    else:
        self._goal_fails += 1
    if self._goal_fails >= GOAL_FAIL_MAX or self._goal_age >= GOAL_AGE_MAX:
        self._goal = None            # invalida → re-pergunta quando o cooldown permitir
# (3) fallback: navigate → plan → greedy
if cand is None and self._nav_on:
    nk = navigate(scene, self._move)
    if nk is not None: cand = keymap.get(nk)
if cand is None and self._plan_on and cands:
    pk = plan(state_signature(scene), [c.key for c in cands],
              self._tmodel, self._novelty, self._novelty.goal_anchors)
    if pk is not None: cand = keymap.get(pk)
if cand is None:
    cand = self._policy.decide(...)  # greedy v7
```

**Chamadas esparsas garantidas:** o LLM só é consultado quando não há meta e
passou o cooldown → no máximo ~1 chamada a cada `QUERY_COOLDOWN` passos, e
tipicamente 1 por nível (a meta persiste até cumprir/invalidar). Isso cabe no
orçamento de 9h.

## Erros e casos de borda

- **`CAUSAL_LLM` off / `NullLLMClient`:** nunca consulta / resposta `""` →
  `parse_goal` None → sem meta → pilha determinística (v9/v8/v7). Idêntico ao
  v10 pré-LLM.
- **Meta alucinada/invável:** `execute_goal` None → conta falha → invalida após
  `GOAL_FAIL_MAX` → re-pergunta. Nunca trava.
- **Meta boa mas jogo não completa:** `GOAL_AGE_MAX` invalida e re-pergunta com a
  cena/dinâmica novas.
- **Nível sobe:** meta zerada → nova consulta pro próximo nível.
- **Determinismo:** com `FakeLLM` fixo, o comportamento é determinístico.

## Testes (TDD, `tests/causal/`)

`navigate.py` (do v9): `test_navigate.py` (MovementModel/_moved_object) +
`test_navigate_path.py` (navigate). (Casos idênticos ao plano v9.)

Controlador (`test_agent_llm.py`), com `FakeLLM`:
1. **usa a meta do LLM:** `CAUSAL_LLM=1` + `FakeLLM('{"type":"press","action":"ACTION1"}')`
   e `ACTION1` disponível → a ação escolhida é `ACTION1`; `self._goal` setado.
2. **resposta inválida → fallback:** `FakeLLM("lixo")` → `self._goal` None; não
   estoura; retorna uma ação.
3. **invalidação por falha:** meta cujo `execute_goal` dá None por
   `GOAL_FAIL_MAX` passos → `self._goal` volta a None.
4. **esparso:** com uma meta ativa válida, o `FakeLLM` (contador de chamadas) é
   chamado **1×** em vários passos (não re-consulta com meta ativa).
5. **off não consulta:** `CAUSAL_LLM=0` → `FakeLLM` nunca é chamado; `self._goal`
   fica None; roda determinístico.
6. **level-up limpa a meta:** ao subir `levels_completed`, `self._goal` → None.
7. **Regressão:** os 143 testes v1–v10b seguem verdes (com `CAUSAL_LLM` off por
   padrão local, nada muda).

## Fora de escopo

- v10c (sandbox de código); v11/v12.
- Ajuste fino de `QUERY_COOLDOWN`/`GOAL_*` (empírico no Kaggle).

## Critério de pronto

- `navigate.py` implementado; agente com controlador LLM (esparso, validação,
  re-pergunta) e pilha LLM→navigate→plan→greedy; toggles `CAUSAL_LLM`/`CAUSAL_NAV`.
- 143 testes v1–v10b + novos verdes; `CAUSAL_LLM` off por padrão local.
