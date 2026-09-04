# Fix do prompt direct — travar em `available` + formato de resposta condicional

> Design doc. Fonte de verdade: `/home/lkenzo/projetos/safe/CLAUDE.md`. Data: 2026-09-01. Base: `main` 0f24be7.

## Contexto

1º run ao vivo do Lever #2 (32B, offline): **vc33 (click)** `direct_hits 98/99`; **ls20 (teclado)** `direct_hits 3/100`. O direct só funciona de fato em jogos de clique. Causa no ls20: `build_direct_prompt` (a) exibe os nomes de ação no formato bruto (`str(GameAction.ACTION1)`) em vez do `ACTIONk` que o `keymap`/`execute_goal` esperam, e (b) oferta `click_cell` mesmo sem clique disponível → o 32B emite formato incompatível → `keymap.get(key)` = None → miss → cai na pilha antiga (fixação em ACTION2).

Este fix torna o direct **realmente aplicável em jogos de teclado**, pra o próximo teste (Lever #1, modelo melhor) ser justo. **Não** cruza nível sozinho (vc33 a 98% já dá 0 níveis — o gargalo é o modelo).

## Escopo

Contido em `agents/causal/llm.py`: reescrever `build_direct_prompt`. Atualizar os testes em `tests/causal/test_llm_direct.py`. Regenerar os 2 notebooks. Default-safe (só melhora o prompt; nenhuma mudança na pilha de decisão ou em `_direct_decide`).

## Mudança em `build_direct_prompt(scene, dyn, last=None) -> str`

1. **Normalizar nomes:** para cada item de `dyn["available"]`, extrair o token `ACTIONk` via regex `re.compile(r"ACTION\d")` (fallback: o próprio string se não casar). O prompt passa a mostrar/pedir exatamente `ACTIONk` — a forma que `execute_goal`(press) devolve e o `keymap` indexa.
2. **Detectar clique:** `is_click = "ACTION6" in names`. `press = [n for n in names if n != "ACTION6"]`.
3. **Ofertar só o formato compatível:**
   - Se `press` não vazio → oferta a linha `{"type":"press","action":"<press[0]>"}` com nota "action DEVE estar em AVAILABLE_ACTIONS".
   - Se `is_click` → oferta a linha `{"type":"click_cell","gx":0,"gy":0}` (grid 6×6).
   - Um jogo de teclado (sem ACTION6) **não** vê `click_cell`; um jogo de clique-puro (ex. só ACTION6) **não** vê `press` (press vazio).
4. **Travar duro:** a linha `AVAILABLE_ACTIONS: <names>` ganha o texto "responda com UMA acao SO desta lista".

Assinatura e uso (`_direct_decide` chama `build_direct_prompt(scene, dyn, last)`) inalterados. `last` (feedback da última ação) mantém o comportamento atual (linha só quando `last["key"]` existe).

Adicionar `import re` no topo de `llm.py` (ou reusar se já houver).

## Estrutura do prompt (resultante)

```
OBJETOS (N):
  id=.. color=.. centroid=.. size=.. bbox=..
AVAILABLE_ACTIONS: ['ACTION1', 'ACTION2']   (responda com UMA acao SO desta lista)
[linha de feedback da última ação, se houver]
Escolha a PROXIMA acao imediata (uma so). Responda APENAS um JSON, sem markdown, sem prosa, um de:
{"type":"press","action":"ACTION1"}   (action DEVE estar em AVAILABLE_ACTIONS)
[{"type":"click_cell","gx":0,"gy":0}   (gx,gy em 0..5) — SÓ se ACTION6 disponível]
```

## Testes (TDD, `tests/causal/test_llm_direct.py`)

Os testes existentes que assumem "sempre oferece press E click" são atualizados p/ a intenção real (condicional). Novos/atualizados:

1. **`test_direct_prompt_lists_available`** (mantém): available `["ACTION1","ACTION2"]` → `AVAILABLE_ACTIONS` presente com `ACTION1`/`ACTION2`.
2. **`test_direct_prompt_shows_objects`** (mantém): `color=3` no prompt.
3. **`test_direct_prompt_last_feedback_present`** / **`_omitted_when_none`** (mantêm).
4. **`test_direct_prompt_keyboard_no_click`** (novo): available `["ACTION1","ACTION2"]` (sem ACTION6) → `'"type":"press"'` presente **e** `'"type":"click_cell"'` **ausente**.
5. **`test_direct_prompt_click_offers_click`** (substitui o antigo `_asks_single_action`): available `["ACTION6"]` → `'"type":"click_cell"'` presente; press ausente (lista sem tecla).
6. **`test_direct_prompt_normalizes_action_names`** (novo): available `["GameAction.ACTION1"]` (forma bruta) → prompt contém `ACTION1` mas **não** `GameAction.ACTION1`.
7. **`test_direct_prompt_hard_constrains`** (novo): prompt contém a instrução "SO desta lista" (trava dura).

## Empacotamento

Regenerar `kaggle/submission.ipynb` e `kaggle/offline.ipynb` (o `build_direct_prompt` novo entra no `llm.py` embarcado); verificar o embed. Nenhuma mudança de env/toggle.

## Validação real (fora daqui)

Próximo run offline (idealmente já com o modelo do Lever #1): observar o `direct_hits` do **ls20/jogos de teclado** subir de ~3% pra perto de 100% (o direct passa a aplicar de fato). O teste de nível segue dependendo do modelo.
