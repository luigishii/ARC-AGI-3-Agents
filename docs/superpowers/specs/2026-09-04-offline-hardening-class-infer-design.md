# Offline hardening + inferência de classe por LLM (jogo não-visto)

> Design doc. Fonte de verdade: `/home/lkenzo/projetos/safe/CLAUDE.md`. Data: 2026-09-04. Base: `main` 7b231fe.

## Contexto

`swarm.py` registra que vc33/tn36/lp85 cruzaram 1 nível offline — mas via `_GAME_KNOWLEDGE`
(tabela por `game_id`), que vale zero na eval privada (jogos não-vistos). Além disso, com
gpt-oss-120b (~30s/chamada) a config offline (`DEFER=50`, `MAX_CALLS=3`, timeout 120s) fazia
o LLM decidir ~3 ações/jogo e estourar o timeout no meio, com a thread daemon continuando a
rodar junto do próximo jogo. E o path do modelo era hardcoded: slug errado → `NullLLMClient`
silencioso.

## Mudanças (todas default-safe; os builders ligam os flags)

1. **Descoberta do path do modelo** (`build_notebook.MODEL_DISCOVERY`, usado nos 2 notebooks):
   se `MODEL_DATASET_PATH` não existe, glob de `config.json` com `gpt-oss` no path (mais raso
   vence); grava `QWEN_MODEL_PATH` por último no `.env` (dotenv: última vence) e imprime OK/NÃO EXISTE.
2. **Timeout cooperativo**: `Agent.stop_requested` (base) checado no loop `main`; o `Swarm`
   sequencial seta a flag no timeout e faz um 2º `join`. Offline `SWARM_GAME_TIMEOUT=600`.
3. **Inferência de classe** (`CAUSAL_CLASS=1`, `CAUSAL_CLASS_AT=8`): quando `_gk` está vazio
   (jogo desconhecido ou modo cego), 1 chamada de LLM com a taxonomia A–F de `docs/GAMES.md`
   + grid ASCII + objetos + available → `parse_class` → preenche os mesmos slots da tabela
   (`cls/avatar/target/click/hud_rows/hud_cols`) e re-seeda o HUD. Não espera `LLM_DEFER`.
   Resposta inválida → desiste sem retry. Diag `gk_src` = `table:X` | `llm:X` | `None`.
4. **Modo cego** `CAUSAL_GK=0`: ignora a tabela → o run offline que reflete a eval privada.
   Editável na célula 1 do `offline.ipynb`.
5. **Fix do prompt direct** (spec 2026-09-01, agora implementada): nomes normalizados p/
   `ACTIONk`, trava dura na lista, oferta `press` só se há tecla e `click_cell` só se ACTION6.
   Bônus: sob clickmap, `click_cell(gx,gy)` do LLM é mapeado pro candidato de clique que cai
   na célula (antes nunca casava com as chaves `ACTION6@c9s0@gx,gy` → miss).
6. **Higiene**: testes do harness (`tests/unit/test_core.py`, `test_swarm.py`) atualizados
   pro pacote `arc_agi`; worktrees `winframe-fix` (merged) e `direct-prompt-fix` (absorvida)
   removidas.

## Validação real (Kaggle offline)

- Log deve mostrar `QWEN_MODEL_PATH -> ... (OK)` e `[causal] LLM ativo: hf`.
- `phase2_stats.gk_src` = `llm:X` nos jogos desconhecidos / modo cego.
- Comparar `levels_completed` em `CAUSAL_GK=1` vs `CAUSAL_GK=0`: a diferença é o que a
  tabela dá e a eval NÃO dá; o valor em modo cego é a previsão honesta da submissão.

## Adendo (mesma sessão) — pré-submissão com gpt-oss-120b

7. **Cap global de chamadas** `CAUSAL_LLM_TOTAL_CALLS` (`agent._LLM_TOTAL`, por processo,
   todas as threads) + `_take_llm_call(per_game)` em todos os call sites; a classe e os
   loops de reward/f_τ só respeitam o global. Diag `llm_total_calls`.
8. **Deadline global** `SWARM_DEADLINE_S` (desde o início do `Swarm.main`, inclui o load do
   modelo): paralelo → `join(timeout=restante)` e `stop_requested` em todos; sequencial →
   timeout por jogo limitado ao restante e pula jogos após o prazo.
9. **Papéis inferidos no prompt do direct** (`GAME CLASS`/`AVATAR COLOR`/`TARGET COLOR`/
   `CLICKABLE COLORS`).
10. **Budget por classe** em jogo não-visto sem env/tabela (`_CLASS_BUDGETS`, click-only=80).
11. **Esforço por chamada**: `complete(prompt, effort=None)` em todos os clients;
    `resolve_effort`; direct usa `CAUSAL_DIRECT_EFFORT` (default `low`), classe/reward usam
    `CAUSAL_EFFORT`. `agent._complete` só passa `effort=` se o client aceita.
12. **Env da submissão** alinhado ao gpt-oss: `CAUSAL_MAX_ACTIONS=1500` (era 100000; reativa
    early-exit), `CAUSAL_LLM_MAX_CALLS=4`, `CAUSAL_LLM_DEFER=50`, `CAUSAL_DIRECT_COOLDOWN=20`,
    `CAUSAL_LLM_TOTAL_CALLS=600`, `CAUSAL_DIRECT_EFFORT=low`, `SWARM_DEADLINE_S=30000`.

**Validação ao vivo (API pública, sem LLM, 120 ações):** vc33 cruzou o **nível 1** tanto com
a tabela (`gk_src=table:C`, level-up na ação 24) quanto em **modo cego** (`gk_src=None`,
ação 36). O clickmap genérico transfere; o nível 2 não saiu em 120 ações.
