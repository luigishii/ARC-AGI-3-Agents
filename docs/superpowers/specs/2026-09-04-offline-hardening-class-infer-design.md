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
