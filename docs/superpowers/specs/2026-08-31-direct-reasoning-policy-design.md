# Score-max Lever #2 — Política de Raciocínio Direto Passo-a-Passo

> Design doc. Fonte de verdade do projeto: `/home/lkenzo/projetos/safe/CLAUDE.md`.
> Data: 2026-08-31. Base: `main` local 88d3547 (393 testes verdes).

## Contexto & Motivação

O pipeline atual usa o LLM só pra **inferir uma META persistente** (`press`/`click_cell`/`reach`/`code`), esparso (`QUERY_COOLDOWN=8`, ~1 chamada/nível), executada por vários frames até invalidar. Provou-se, em ~15 runs offline com Qwen3-32B real, que isso dá **0 níveis**: sem cruzar 1 nível a reward é inaprendível (problema circular), e a meta persistente fixa/erra.

**Lição Tycho (paper 100% RHAE):** a política vencedora torna o **world-model verificado OPCIONAL** e usa **raciocínio direto** quando basta — pontuou MAIS que a máquina de world-model elaborada. Os líderes cruzam ~6% porque o modelo **olha a cena e escolhe o próximo passo direto**, frame a frame, sem depender de aprender reward.

Este lever adiciona essa política: o LLM escolhe a **próxima ação imediata** a cada frame (gated por cooldown), a partir da cena objeto-cêntrica + `available_actions` + feedback do efeito da última ação, replanejando continuamente. A fundação causal valida e executa.

**Objetivo commitado:** score-max (não o world-model principiado). Ver o pivot no CLAUDE.md (31/ago).

## Escopo

Aditivo, contido em `agents/causal/llm.py` (+1 função) e `agents/causal/agent.py` (constantes, init, 1 método, wiring, stats). Novo toggle `CAUSAL_DIRECT` (default **off** → zero regressão). Sem novos módulos. Suíte segue verde.

**Fora de escopo:** trocar o modelo (Lever #1, ação do usuário); amostragem massiva por-frame (Lever #3); mexer nas camadas existentes (navigate/IW/rprog/η/cover) além do gate condicional.

## Arquitetura

### Componente 1 — `llm.py: build_direct_prompt(scene, dyn, last)`

Novo builder de prompt orientado a AÇÃO (distinto de `build_prompt`, orientado a META). Serializa:

- **Objetos:** `id`, `color`, `centroid`, `size` (mesma serialização de `build_prompt`).
- **`AVAILABLE_ACTIONS`:** a lista `dyn["available"]` com "use SÓ essas".
- **Feedback da última ação** (`last`): se houver, linha `"Sua última acao <key> produziu: <efeito|nenhuma mudanca>. Escolha a PROXIMA acao que faz PROGRESSO; NAO repita uma acao que nao mudou nada."`. Se `last` é `None` ou `last["key"]` é `None` (1º passo), omite a linha.
- **Instrução de resposta:** JSON de UMA ação imediata — `{"type":"press","action":"ACTIONk"}` ou `{"type":"click_cell","gx":<0-5>,"gy":<0-5>}` (grid 6×6 de clique, convenção do `candidates()`/`execute_goal`). Sem markdown, sem prosa.

`last` é `{"key": str|None, "effect": str|None}` ou `None`. O parsing reusa `parse_goal` (robusto a cercas/prosa, valida `press`/`click_cell`); a execução reusa `execute_goal` (`press`→action, `click_cell`→`ACTION6@cell=gx,gy`).

**Assinatura:** `build_direct_prompt(scene, dyn: dict, last: dict | None) -> str`.

### Componente 2 — `agent.py: _direct_decide(scene, avail, cands, keymap, moves)`

Novo método. Retorna um `Candidate` (o tipo que `keymap` guarda) ou `None`.

```
if self._since_direct < self._direct_cooldown: return None
if self._llm_calls >= self._llm_max: return None
self._since_direct = 0
self._llm_calls += 1
self._direct_calls += 1
dyn = {"available": [str(a) for a in avail], "moves": moves, "notes": ""}
last = {"key": self._last_key, "effect": self._last_effect_kind}
try:
    resp = self._llm.complete(build_direct_prompt(scene, dyn, last))
    g = parse_goal(resp)
    key = execute_goal(g, scene, moves) if g is not None else None
except Exception:
    return None
cand = keymap.get(key) if key is not None else None
if cand is not None:
    self._direct_hits += 1
return cand
```

- **1 completion por query** (per-frame; custo importa — não usa `complete_many`).
- **available-guard** natural: só retorna cand se a key emitida ∈ `keymap` (derivado de `avail`).
- **Exception-safe:** qualquer falha (JSON inválido, ação indisponível, cliente Null) → `None` → cai na pilha determinística.
- Consome o budget compartilhado `_llm_calls`/`_llm_max`.

O caller (`choose_action`) só chama `_direct_decide` quando `self._direct_on` é True e `cand is None` — o gate de `_direct_on`/`_since_direct += 1` fica na `choose_action` (ver Componente 3).

### Componente 3 — Wiring na `choose_action`

1. **Feedback da última ação:** no fecha-loop (após `actual = self._model.observe(...)`, ~agent.py:218), na transição decisão→decisão gravar `self._last_effect_kind = actual.kind`. No `full_reset`/`need_reset` (limpeza no topo da `choose_action`) e no level-up, resetar `self._last_effect_kind = None`.
2. **Incremento do cooldown:** junto aos outros (`self._since_query += 1`), adicionar `self._since_direct += 1`.
3. **Gate do bloco de consulta-meta persistente (agent.py:255-256):** quando `CAUSAL_DIRECT` on, **pular** esse bloco — direct substitui a inferência-de-meta como driver do LLM (evita gastar budget 2× e conflito). Concretamente: adicionar `and not self._direct_on` à condição existente. O bloco `1b` (síntese de `f_τ`/reward, agent.py:277-285) **permanece intacto**.
4. **Camada direct no topo da pilha:** com `cand` inicializado como `None` antes do goal-path (agent.py:286), inserir a PRIMEIRA tentativa:
   ```
   if cand is None and self._direct_on:
       cand = self._direct_decide(scene, avail, cands, keymap, moves)
   ```
   O restante (goal→navigate→rprog→IW→plan→η→cover→greedy) e o `_antifix` final ficam inalterados — o antifix segue guardando ações repetidas do direct.

### Componente 4 — Diagnóstico & empacotamento

- `phase2_stats()` ganha `"direct_calls"` e `"direct_hits"`.
- `CAUSAL_DIRECT=1` no `.env` gerado pelos 2 builders (`kaggle/build_notebook.py`, `kaggle/build_offline_notebook.py`).
- Notebooks regenerados; embed base64 verificado (a nova função vive em `llm.py`, já embarcado).

## Constantes

- `DIRECT_COOLDOWN` lido de `CAUSAL_DIRECT_COOLDOWN` (default **2**): equilíbrio entre replanejar-cada-frame e caber nas 9h; env-tunável pra 1. Entre frames de cooldown, `_direct_decide` retorna `None` e a pilha determinística decide. Guardado em `self._direct_cooldown`.

## Estado novo no `_init_causal_state`

- `self._direct_on = os.environ.get("CAUSAL_DIRECT", "0") != "0"`
- `self._direct_cooldown = int(os.environ.get("CAUSAL_DIRECT_COOLDOWN", "2"))`
- `self._since_direct = 10 ** 9` (permite consulta no 1º frame elegível, como `_since_query`)
- `self._last_effect_kind = None`
- `self._direct_calls = 0`, `self._direct_hits = 0`

## Fluxo por frame (direct on)

fecha-loop (grava `_last_effect_kind`) → `_since_direct += 1` → se cooldown+budget: `build_direct_prompt(scene, last-effect)` → LLM (1 completion) → `parse_goal` → `execute_goal` → key ∈ keymap? → cand → `_antifix`. Miss → cai em navigate→rprog→IW→…→greedy → `_antifix`. Bloco de meta-persistente **pulado**; síntese de `f_τ`/reward mantida.

## Tratamento de erro

Toda falha do LLM/parse/execução vira `None` (fall-through determinístico). `NullLLMClient` (LLM não subiu) → `complete` devolve `""` → `parse_goal` → `None` → fall-through. Nenhuma exceção propaga da `_direct_decide`.

## Testes (TDD, FakeLLM)

`FakeLLM` = cliente com fila de respostas (padrão dos testes existentes, ex. `test_agent_llm.py`). Estados via helper `_scene(coords)` (numpy grids) e `CausalObjectAgent.__new__` + `_init_causal_state()` com `monkeypatch` dos envs. `FrameData`-like via os stubs já usados nos testes de agente.

1. **Direct on + ação válida disponível** → `FakeLLM` devolve `{"type":"press","action":"ACTION1"}`, `ACTION1 ∈ avail` → `choose_action` retorna ACTION1; `phase2_stats()["direct_hits"] == 1`.
2. **Direct on + ação indisponível/garbage** → `FakeLLM` devolve ação fora de `avail` (ou lixo) → `_direct_decide` → `None` → decisão vem da pilha determinística (não crasha; retorna uma ação válida).
3. **Direct off** → `CAUSAL_DIRECT` unset → `_direct_decide` nunca chamado (`direct_calls == 0`); goal-path inalterado.
4. **Cooldown** → `CAUSAL_DIRECT_COOLDOWN=2`; a partir de `_since_direct` alto, 1ª chamada consulta e zera; 2º frame consecutivo (`_since_direct==1 < 2`) NÃO consulta → `direct_calls == 1` após os 2 frames.
5. **Budget esgotado** → `CAUSAL_LLM_MAX_CALLS=0` → nenhuma consulta direct (`direct_calls == 0`).
6. **Direct on ⇒ meta-persistente pulada** → com direct on e `_since_query` alto, o bloco 255-273 não seta `self._goal` (segue `None`).
7. **click_cell** → `FakeLLM` devolve `{"type":"click_cell","gx":2,"gy":3}` com `ACTION6 ∈ avail` → key `ACTION6@cell=2,3` ∈ keymap → cand retornado (`direct_hits == 1`).

Suíte inteira (393 + novos) verde ao fim.

## Validação real (fora deste ambiente)

Só o Kaggle offline (RTX Pro 6000, 32B real, internet OFF) valida o comportamento: regenerar `offline.ipynb`, rodar 1-2 jogos com `CAUSAL_DIRECT=1`, observar `direct_calls>0`/`direct_hits>0`, **ações variadas** (não fixação) e o teste decisivo `levels_completed`. Aqui só o encanamento é testável (FakeLLM). Ressalva honesta: mesmo funcionando, o teto realista é cruzar o 1º nível em alguns jogos = algumas décimas acima de 0.08.

## Anti-regressão

Default-off no toggle novo → o caminho existente (goal-inference + pilha) roda idêntico quando `CAUSAL_DIRECT` unset. Uma alavanca por vez: nenhuma mudança nas camadas navigate/IW/rprog/η/cover além do gate condicional do bloco de meta.
