# Grounding de papel do avatar na reward — design

**Data:** 2026-08-31
**Status:** spec aprovada

## Problema (provado no run offline 32B, ls20+ka59)

O lever anterior (reward espacial) funcionou: `reward_src` virou distância de Manhattan em
vez de contagem-de-cor. MAS ainda 0 níveis, `iw_goal_hits=0`, `reward_real_true=0/192`.
Causa isolada: a reward está correta na FORMA mas **mira o objeto errado** — o LLM chuta
`pts[0]` como avatar e escolhe o alvo por cor arbitrária, sem grounding. Exemplos reais:
- ls20: `a = pts[0]`, alvo = objetos cor 3 (mas o avatar decodificado é o bloco vermelho cor 9).
- ka59: `a = pts[0]`, alvo = cor 15 (mas o mover decodificado é o marcador 3×3).

`navigate.MovementModel` (agent.py:86, `self._move`) **já aprende qual objeto é o avatar**:
`avatar_id()` (navigate.py:41) retorna o `o.id` que mais se moveu com as ações (ou `None`
no cold-start). Esse conhecimento não chega ao prompt de síntese da reward.

## Solução

Injetar o avatar aprendido no `_build_reward_prompt`: mapear `avatar_id()` → índice no
`state` e dizer ao LLM "o objeto controlável é `state[K]`", com os few-shot usando `pts[K]`
como o objeto móvel. Ancora metade do par (avatar), corrigindo o erro mais grosseiro
(`pts[0]` arbitrário).

### Componente único: `_build_reward_prompt` (agents/causal/agent.py)

Fluxo novo dentro do método:
1. `aid = self._move.avatar_id()` — `o.id` do avatar aprendido, ou `None`.
2. Mapear pro índice do `state`: iterar `list(scene.objects)[:8]` e achar `i` onde
   `o.id == aid` → `avatar_idx`. (A ordem de `scene.objects` é a mesma usada p/ montar o
   `state` em `_try_learn_reward` — verificado.)
3. **Se `avatar_idx` encontrado:** incluir no prompt a linha de grounding
   `"OBJETO CONTROLAVEL (avatar) = state[K]; a reward DEVE medir a distancia DELE (state[K]) ate o alvo."`
   e os few-shot usam `pts[K]` (não `pts[0]`).
4. **Se `None` / avatar fora do top-8:** fallback ao comportamento atual (few-shot com
   `pts[0]`, sem linha de grounding). Cold-start-safe.

### Contrato do `state` e demais componentes (inalterados)

`_spatial_context` permanece puro (objects→texto). `state`, `accept_reward`,
`static_reward_check`, `compile_reward` e a pilha de decisão não mudam. Default-safe
(mesmo caminho já sob `CAUSAL_LLM`, sem toggle novo). `self._move` já é populado no
close-loop (`_move.observe`, agent.py:205).

## Arquitetura / isolamento

- Mudança contida a **`_build_reward_prompt`**. Consome `self._move.avatar_id()` (já existe)
  e `scene.objects` (já disponível). Um helper interno pode calcular o índice, mas cabe
  inline no método (poucas linhas).
- `_spatial_context` não muda.

## Testes (offline, FakeLLM + MovementModel primado)

Para primar o avatar: setar `a._move.avatar_counts = {oid: 1}` diretamente (determinístico),
com `oid` = `o.id` de um objeto conhecido da cena de teste; então `a._move.avatar_id()`
retorna esse `oid`.

1. Avatar mapeado pro índice K → `_build_reward_prompt` contém `state[K]` e o few-shot
   referencia `pts[K]`.
2. `avatar_id()` retorna `None` (nenhum movimento observado) → prompt **sem** linha de
   grounding; few-shot cai em `pts[0]`.
3. Regressão: uma reward de distância usando o índice grounded **passa**
   `static_reward_check` + `accept_reward` (gradiente real). Reward constante continua
   **rejeitada** (hardening intacto).

## Fora de escopo (YAGNI / follow-up)

- **Grounding do ALVO** — o alvo ainda é inferido pelo LLM. Se o alvo errado seguir sendo o
  gargalo no próximo run, vira o lever seguinte.
- Afrouxar `goal_flag=(dist==0)` (preciso demais) — separado; observar se persiste.

## Limitação honesta

Ancora só o avatar (metade do par). O alvo continua chutado. Mas corrige o erro dominante
visto no run (avatar arbitrário `pts[0]`). Cruzar nível de fato só se confirma no Kaggle.

## Validação

Offline não confirma (é prompt/wiring); a validação real é o run offline no Kaggle (32B)
observando `reward_src` referenciar o avatar aprendido (índice consistente com o objeto que
se move), `iw_goal_hits > 0`, e o decisivo `levels_completed > 0`.
