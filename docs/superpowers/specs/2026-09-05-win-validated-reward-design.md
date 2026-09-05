# Lever "reward validada pela vitória" — design (2026-09-05)

## Problema (medido, baseline cego 05/set, `CAUSAL_LLM=0 CAUSAL_GK=0`)

Os jogos que cruzam o L1 (vc33, tn36, lp85, lf52) param no L2 com 100+ ações sobrando. Três fatos, extraídos das gravações com `scratchpad/win_reward_probe.py`:

1. **A reward certa já existia.** No L0 do vc33 e do tn36 a `grounded_multi_reward_fn` tem o frame vencedor como argmax de todos os frames do nível (rank 1/39 e 1/89) e correlação de Spearman tempo×valor de +0.83 e +0.93. Ela *explica* a vitória.
2. **Ela nunca decide.** Pilha atual: sweep → replay → **2phase** → navigate → **rprog** → burst → IW → … Em jogo click-only onde todo botão tem efeito, o `2phase` dispara todo passo ("outro objeto produtivo, o menos visitado") e faz round-robin: vc33 = 144/200 ações no `2phase`, 29/29/28/27 cliques nos 4 botões. O `rprog` (sobe a reward real) fica faminto.
3. **Pós-L1 a reward vira ruído.** O level-up troca a reward pela `template(win-grid)` ("o objeto está onde o grid vencedor do L1 tinha essa cor?"). O layout do L2 é outro. No L2 do vc33 a `multi-align` tem 5 valores distintos e Spearman +0.46 que ninguém sobe.

## Objetivo

Depois da 1ª vitória de um jogo, usar a trajetória rotulada (frames do nível + frame vencedor) para **escolher a reward que explica a vitória** e **deixar essa reward dirigir** o nível seguinte. Métrica: `levels_completed` nos 4 jogos com L1 (cruzar L2 em ≥1) sem regressão nos 25.

## Não-objetivos

- Não mexe no L0 (antes de qualquer vitória a pilha é a atual). Experimento futuro separado.
- Não cria rewards novas; só seleciona entre as existentes em `goals.py`.
- Não toca em LLM/Kaggle além do flag nos builders.

## Design

### Componente 1 — `agents/causal/winselect.py` (puro, numpy/stdlib)

- `explain_score(values: list[float], win_idx: int) -> tuple[bool, float]`
  Retorna `(is_top, rho)`: `is_top` = valor no `win_idx` é ≥ todos os outros (empate permitido); `rho` = Spearman entre índice temporal e valor sobre todos os frames (NaN → 0.0).
- `select_win_reward(candidates: list[tuple[str, callable]], level_states: list, win_state) -> tuple[str, callable, float] | None`
  Para cada candidato: avalia `fn(state)[0]` em `level_states + [win_state]` (exceção ou não-finito → descarta); exige `is_top` **e** `rho > 0` **e** ≥2 valores distintos; ordena por `rho` desc; devolve `(nome, fn, rho)` do melhor ou `None`. Com `len(level_states) < 3` devolve `None` (sem evidência).

### Componente 2 — `agent.py` (sob `CAUSAL_WINREWARD=1`, default off)

- Estado novo: `_level_states` (lista de estados `[(shape_hash, _obj_state)]` do nível corrente, cap 400, alimentada na transição decisão→decisão, limpa no level-up) e `_win_reward` (`None` ou `(nome, fn, rho)`).
- **Level-up:** monta candidatos `[("multi-align", grounded_multi_reward_fn()), ("pattern", grounded_pattern_reward_fn()), ("pair", grounded_pair_reward_fn()), ("count", grounded_count_reward_fn()), ("diversity", grounded_diversity_reward_fn())]` + `("nav", grounded_reward_fn(avatar, alvo))` quando `avatar_id()` e `_pick_target` resolvem na `win_scene`. Chama `select_win_reward(cands, _level_states, win_state)`. Guarda em `_win_reward`. Limpa `_level_states`.
- **Síntese (`_try_learn_reward`, ramo grounded):** antes do ramo `template`, se `_win_reward` existe → adota `fn`, `reward_src = f"win:{nome}(rho={rho:.2f})"`, retorna. O `template` segue como fallback quando `_win_reward is None`.
- **Pilha:** se `_win_reward is not None` e `_rprog_on`, chama `_rprog_decide(cands, uncapped=True)` **antes** do bloco `2phase` (logo após `replay`); acerto → `layer="rprog"`. O `2phase` e o resto seguem como fallback quando não há Δ médio positivo (≥3 observações por chave). `_antifix` continua ativo no fim.
- `_rprog_decide(cands, uncapped=False)`: `uncapped=True` ignora `_rprog_max`.
- Telemetria: `phase2_stats["win_reward"] = nome|None`, `["win_rho"]`.
- Flag `CAUSAL_WINREWARD=1` nos 2 builders (`build_notebook.py` `.env`; o offline reusa).

### Fluxo

```
nível N: decisões → _level_states acumula
level-up → win_scene → select_win_reward → _win_reward
nível N+1: _try_learn_reward adota win reward → rprog (acima do 2phase) sobe Δ real
           sem sinal → 2phase/…/greedy (inalterado)
```

### Erros

- Reward candidata que levanta exceção ou devolve não-finito em qualquer estado → descartada.
- Nenhuma válida → `_win_reward=None` → comportamento atual (template).
- Flag off → caminho byte-idêntico (nenhuma chamada nova).

## Testes (TDD)

1. `winselect`: (a) entre 3 candidatos sintéticos, escolhe o que é argmax no win e crescente; (b) candidato com exceção descartado; (c) argmax mas rho≤0 → inválido; (d) < 3 estados → None; (e) constante → inválido.
2. `agent` level-up com flag: trajetória sintética onde 2 objetos da mesma cor convergem até o win → `reward_src` começa com `win:multi-align`; sem flag → `grounded:template(win-grid)`.
3. `agent` pilha: com `_win_reward` setado, `_rprog` com ≥3 Δ>0 numa chave e última ação com efeito (2phase dispararia), a `reasoning["layer"]` da decisão é `rprog` e a key é a de maior Δ; sem flag → `2phase`.
4. `uncapped`: `_rprog_fires >= _rprog_max` não bloqueia com `uncapped=True`.
5. Builders: `.env` contém `CAUSAL_WINREWARD=1`.

## Validação real

Loop local grátis (API pública, `CAUSAL_LLM=0 CAUSAL_GK=0 CAUSAL_WINREWARD=1`, 200 ações): vc33, tn36, lp85, lf52 → `win_reward`, `rprog_fires`, `layers`, `levels_completed`. Depois os 25 (sem regressão de L1).
