# Alvo por heurística na reward — design

**Data:** 2026-09-01
**Status:** spec aprovada (heurística: cor rara + compacto)

## Problema (run offline 32B, ls20+ka59, 01/set)

Reward espacial + avatar grounding: mesmo assim `reward_src` mira o **alvo errado** — ls20
alvo = cor 3 (corredor), ka59 = cor 15 (barra conectora); ambos escolhidos pelo LLM por cor
arbitrária. `reward_real_true=0`, `iw_goal_hits=0`. O avatar dá pra aprender do movimento
(`MovementModel`), mas o **alvo** o LLM chuta. Precisamos dar ao prompt um **alvo provável**
principiado, e um few-shot **explícito avatar→alvo** (no run o LLM recaiu em `pts[0]`).

## Solução

No `_build_reward_prompt` (que já computa `avatar_idx`), computar `target_idx` por heurística
e injetar "ALVO PROVAVEL = state[T]" + few-shot explícito `a=pts[AVATAR]; t=pts[TARGET]`.

### Heurística do alvo (cor rara + compacto)

Sobre `list(scene.objects)[:8]`, com `avatar_idx` conhecido:
1. **candidatos** = todos os índices exceto `avatar_idx`.
2. **excluir o maior objeto** (maior `size` — provável fundo).
3. **excluir barras/HUD alongadas**: para `bbox=(r0,c0,r1,c1)`, `w=c1-c0+1`, `h=r1-r0+1`;
   alongado se `max(w,h) / max(1,min(w,h)) >= 4`.
4. entre os candidatos restantes, escolher a **cor mais rara** — menor contagem daquela cor
   entre TODOS os objetos da cena; desempate: **menor `size`**; depois **menor distância
   Manhattan ao avatar** (centróides).
5. **fallback:** se as exclusões zerarem os candidatos, usar o **não-avatar mais próximo do
   avatar**; se ainda vazio (só há o avatar), sem hint de alvo.

### Componente único: `_build_reward_prompt` (agents/causal/agent.py)

- Já tem `avatar_idx` (ou `None`). Se `avatar_idx is not None`:
  - `target_idx = _pick_target(scene.objects, avatar_idx)` (helper puro no módulo).
  - se `target_idx is not None`: injetar
    `ALVO PROVAVEL = state[T]; a reward deve medir a distancia state[K] -> state[T].`
    e o few-shot de distância vira **explícito**:
    `a=pts[K]; t=pts[T]; d=abs(a['x']-t['x'])+abs(a['y']-t['y']); return (-float(d), d==0)`.
- Se `avatar_idx is None` (cold-start / jogo de clique): comportamento atual (few-shot com
  `pts[0]`, sem hints). Cold-start-safe.

### Contrato / demais componentes (inalterados)

`_spatial_context` puro, `state`, `accept_reward`, `static_reward_check`, pilha de decisão:
sem mudança. Default-safe (mesmo caminho sob `CAUSAL_LLM`). Novo helper `_pick_target` no
nível de módulo (puro, testável isolado).

## Testes (offline, FakeLLM + estados sintéticos)

1. `_pick_target`: cena com avatar + barra alongada (color A) + objeto compacto de cor rara
   (color B, 1 ocorrência) + vários de cor comum → retorna o índice do objeto B (raro,
   compacto), **não** a barra nem o comum.
2. `_pick_target`: a barra alongada é excluída mesmo se for a cor mais rara.
3. `_build_reward_prompt` com avatar primado (via `_move.avatar_counts`) → prompt contém
   `ALVO PROVAVEL = state[T]` e few-shot `pts[K]`/`pts[T]` (K=avatar, T=alvo).
4. Sem avatar (`avatar_counts` vazio) → sem `ALVO PROVAVEL`, fallback `pts[0]`.
5. Regressão: reward de distância avatar→alvo **aceita** (`accept_reward`); constante
   **rejeitada** (hardening intacto).

## Fora de escopo (YAGNI)

- Casamento de padrão (sc25) — a heurística é single-target; jogos de arranjo ficam de fora.
- Aprender o alvo de um level-up real — é o que não temos (problema circular). Esta heurística
  é a alternativa possível sem sinal de sucesso.

## Limitação honesta

Ainda é chute — principiado (raro/compacto/estático ≈ "objetivo" na intuição ARC), mas chute.
Se o alvo real não for saliente/compacto, erra. É a **última cartada da rota de reward**: se
o próximo run offline ainda der 0 níveis, a evidência fecha o caso a favor de consolidar/pivotar.

## Validação

Offline (32B, ls20+ka59): observar `reward_src` medir distância avatar→**alvo compacto/raro**
(não mais cor de corredor/barra), `iw_goal_hits>0`, e o decisivo `levels_completed>0`.
