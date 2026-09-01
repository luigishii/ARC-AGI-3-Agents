# Consertar aprendizado do avatar (upstream) — design

**Data:** 2026-09-01
**Status:** spec aprovada

## Problema (byte-provado em 3 runs offline)

Os levers "avatar grounding" e "alvo por heurística" saíram **byte-a-byte idênticos** ao run
sem eles → greedy: prompt idêntico → **nunca entraram** (fallback `pts[0]`). Causa raiz:
- `MovementModel.observe` só popula `avatar_counts` quando `_moved_object(prev,curr)` retorna
  um objeto; `_moved_object` retorna `None` se **≠1** objeto moveu.
- Em ls20/ka59 a **barra de HUD encolhe TODA ação** (muda `size`/centróide) junto com o avatar
  → sempre ≥2 objetos "movem" → `_moved_object=None` → **avatar nunca aprendido** →
  `avatar_id()=None` → grounding/alvo caem no fallback.
- Além disso a reward sintetiza em `_since_type>=TYPE_COOLDOWN=8` (~ação 8), 1×, e congela —
  pode ser antes de o avatar ser aprendido.

## Solução (2 componentes)

### Componente 1 — `_moved_object` isola o mover RÍGIDO (`agents/causal/navigate.py`)

Entre os objetos cujo centróide mudou, o **avatar transla rigidamente** (forma+tamanho
preservados), enquanto a **barra de HUD encolhe** (muda `size`/`shape_hash`). Filtrar os
**rígidos**: `curr.shape_hash == prev.shape_hash AND curr.size == prev.size`.
- exatamente **1 rígido** → retorna `(id, (dr,dc))` dele;
- 0 ou ≥2 rígidos → `None` (ambíguo, como hoje).

Isso destrava o aprendizado do avatar em jogos com HUD que deplета.

### Componente 2 — adiar síntese da reward até avatar conhecido (`agents/causal/agent.py`)

Gate atual (linha ~281): `if self._reward_fn is None: self._try_learn_reward(scene)`.
Novo:
```
if self._reward_fn is None:
    if self._move.avatar_id() is not None or self._reward_defer >= REWARD_DEFER_MAX:
        self._try_learn_reward(scene)
    else:
        self._reward_defer += 1
```
- `self._reward_defer = 0` init em `_init_causal_state`; constante módulo `REWARD_DEFER_MAX = 4`
  (~4 ticks elegíveis ≈ 32 ações).
- **Cold-start-safe:** jogos de clique (avatar nunca aprendido) sintetizam após o deadline —
  comportamento atual preservado. Só atrasa a síntese o suficiente pra o avatar aparecer.

## Isolamento / componentes inalterados

- Comp.1 = `navigate.py:_moved_object` (função pura). Comp.2 = o gate de reward em `agent.py`.
- Sem mudança em `_build_reward_prompt`, `_pick_target`, `state`, `accept_reward`, pilha.
- Default-safe (mesmo caminho sob `CAUSAL_LLM`; sem toggle novo).

## Testes (offline, numpy/stubs, sem LLM)

1. `_moved_object`: prev/curr com 2 objetos movidos — um **rígido** (size+shape iguais) e um
   **encolhendo** (barra, size menor) → retorna o rígido.
2. `_moved_object`: 2 movers rígidos → `None`. 0 movers → `None`. 1 mover rígido simples →
   retorna ele (regressão do caso feliz atual).
3. `MovementModel.observe` + `avatar_id()`: após observar transições com HUD-ruído, o avatar
   (rígido) é aprendido (`avatar_id()` = id do rígido).
4. Reward-defer (agent, FakeLLM): sem avatar aprendido → `_try_learn_reward` **não** dispara e
   `_reward_defer` incrementa; ao atingir `REWARD_DEFER_MAX` → dispara; com avatar aprendido →
   dispara imediatamente.

## Validação real

Run offline 32B (ls20+ka59): observar `reward_src` **finalmente** referenciar o avatar
aprendido + `ALVO PROVAVEL` (não mais `pts[0]`/cor de corredor), `iw_goal_hits>0`, e o
decisivo `levels_completed>0`.

## Limitação honesta

Destrava o teste real do avatar+alvo grounding (que nunca rodou). Não garante cruzar nível —
faz os 2 levers já construídos finalmente rodarem. Se ainda 0 níveis, a evidência fecha a
rota de reward sintetizada → consolidar/pivotar.
