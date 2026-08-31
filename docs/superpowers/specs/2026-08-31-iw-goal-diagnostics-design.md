# IW Goal-Directed Diagnostics — Design Spec

> **Data:** 2026-08-31 · **Status:** aprovado · **Escopo:** instrumentação (telemetria), sem mudança de comportamento.

## Problema

O `CausalObjectAgent` (Fase-2, com Qwen3-32B) roda o pipeline completo ao vivo — chama o LLM,
sintetiza regras `f_τ`, aprende a `reward_function`, planeja com IW goal-directed — mas
**completa 0 níveis**. A `reward_fn` aprendida **já está ligada** ao IW (`agent.py:309-310`:
`goal_fn_from_reward(self._reward_fn)` passada a `iw_plan`), então o gargalo NÃO é "faltou
wiring". É qualidade: uma de três causas, hoje **invisíveis** no log:

- **(a)** a `reward_fn` nunca é aprendida (Qwen não produz predicado válido);
- **(b)** o predicado aprendido está **errado** (falsos positivos / falsos negativos);
- **(c)** o predicado é correto mas **inalcançável** — o IW sobre o `TypedWorldModel` (`f_τ`)
  nunca acha um caminho até ele (regras incompletas, `max_nodes=300` raso, ou meta longe demais).

Os `phase2_stats` atuais (`llm_kind, llm_calls, n_types, n_rules, reward_learned, eta_rows`)
mostram que a recompensa foi aprendida (`reward_learned=True`) mas **não** distinguem (a)/(b)/(c).

## Objetivo

Tornar as três causas **observáveis** via `phase2_stats` (impresso no log + gravado em
`/kaggle/working/causal_phase2.json`), sem alterar nenhuma decisão do agente. Puro telemetria.
O **conserto** da causa real é uma sessão-follow-up após o run offline (uma alavanca atribuível).

## Escopo

**Dentro:** 5 novas chaves em `phase2_stats`, 4 contadores + 1 campo em `_init_causal_state`,
contagem dentro de `_iw_decide`, avaliação da `reward_fn` na cena real por passo. Só `agents/causal/agent.py` + testes.

**Fora:** qualquer mudança de comportamento (pilha de decisão, IW, síntese, prompts); mudanças
em `iw.py`/`goals.py`; o conserto da causa raiz (depende do que o run revelar).

## Contrato — novas chaves de `phase2_stats`

| Chave | Tipo | Semântica | Isola |
|---|---|---|---|
| `reward_src` | `str \| None` | source exato da `reward_function` sintetizada pelo Qwen | permite **ler** o predicado |
| `iw_goal_calls` | `int` | nº de vezes que `_iw_decide` rodou com `goal_fn` vivo (reward aprendida) | IW goal-directed está disparando? |
| `iw_goal_hits` | `int` | dessas, nº que retornou ação goal-reaching (resultado ≠ None) | IW acha caminho até a meta? |
| `reward_real_true` | `int` | cenas REAIS observadas onde `reward_fn` diz `goal_flag=True` | **(b) vs (c)** |
| `reward_real_evals` | `int` | cenas reais onde a `reward_fn` foi avaliada | denominador de `reward_real_true` |

As chaves existentes permanecem inalteradas.

## Leitura do resultado (para que serve)

- `reward_learned=False` → **(a)**: Qwen nunca produziu predicado válido → mexer no prompt/validação da recompensa.
- aprendida, `iw_goal_calls>0`, `iw_goal_hits=0`, `reward_real_true=0` → **(c)**: predicado nunca satisfeito em estado real/alcançável → predicado estrito demais OU `f_τ` não alcança OU `max_nodes=300` raso.
- `reward_real_true>0` em cenas que claramente **não** são level-up (níveis seguem 0) → **(b)**: predicado errado (falsos positivos) → corrigir síntese/validação da recompensa.

## Mecanismo (tudo em `agent.py`)

1. **Init** em `_init_causal_state`: `self._reward_src = None`, `self._iw_goal_calls = 0`,
   `self._iw_goal_hits = 0`, `self._reward_real_true = 0`, `self._reward_real_evals = 0`.
   (Garante que as chaves existam mesmo sem LLM.)

2. **Contagem no IW** — dentro de `_iw_decide`, quando `gf is not None`:
   incrementa `self._iw_goal_calls`; após `iw_plan(...)`, se o resultado ≠ None,
   incrementa `self._iw_goal_hits`. Retorna o resultado inalterado.
   *(Correto porque `iw_plan` com `goal_fn` retorna não-None **só** num caminho até o goal —
   `iw.py:77`: `return None if goal_fn is not None else best_action`.)*

3. **Avaliação da recompensa na cena real** — quando `self._reward_fn is not None`, uma vez por
   `choose_action` (após montar a `scene`): avalia `goal_fn_from_reward(self._reward_fn)`
   (já à prova de exceção, `goals.py:35-46`) no estado real
   `[(o.shape_hash, _obj_state(o)) for o in scene.objects]`;
   incrementa `reward_real_evals`, e `reward_real_true` se retornar True.

4. **`phase2_stats`** retorna as 5 chaves novas junto das existentes.

`self._reward_src` já é atribuído em `_try_learn_reward` (`agent.py:370`); só falta inicializá-lo
e expô-lo.

## Comportamento e segurança

- **Default-safe:** os contadores só se movem sob `CAUSAL_IW` ligado + recompensa aprendida —
  mesmo gating de hoje. Comportamento idêntico ao atual; nenhuma decisão muda.
- **À prova de exceção:** a avaliação da recompensa reusa `goal_fn_from_reward`, que engole
  exceções do código LLM-autorado (retorna não-goal em falha) → um predicado que quebra não
  derruba o agente nem polui os outros contadores.
- **Sem dependência de GPU/LLM real aqui:** testável 100% com `FakeLLM` neste ambiente.

## Testes (TDD, com `FakeLLM`)

1. `_iw_decide` com `goal_fn` que atinge a meta → `iw_goal_calls==1`, `iw_goal_hits==1`.
2. `_iw_decide` com `goal_fn` que nunca atinge (retorna None) → `iw_goal_calls==1`, `iw_goal_hits==0`.
3. `_iw_decide` sem regras aceitas (retorna None cedo, `gf` nunca montado) → contadores não mexem.
4. avaliação real: `reward_fn` fake com `goal_flag=True` → `reward_real_true` e `reward_real_evals` sobem;
   com `goal_flag=False` → só `reward_real_evals` sobe.
5. `phase2_stats` contém as 5 chaves novas com os valores corretos; `reward_src` presente após aprender.

## Entregável

`agents/causal/agent.py` instrumentado + testes verdes (312 → +5). Notebook offline
(`kaggle/offline.ipynb`) reempacota automaticamente (embute `agent.py`) — o usuário regenera e
roda offline no Kaggle pra ler as 5 chaves. O conserto da causa isolada é a próxima sessão.
