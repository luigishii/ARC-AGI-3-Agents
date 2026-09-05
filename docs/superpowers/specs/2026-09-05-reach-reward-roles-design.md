# Reward de alcance com papéis aprendidos na vitória — design

**Data:** 2026-09-05 · **Status:** aprovado p/ plano · **Toggle:** `CAUSAL_REACH=1` (default off)

## 1. Problema (com dados)

No tn36, a reward validada pela vitória (`win:multi-align`, rho 0.94 no L1) vale **-27
constante nas 112 ações do L2**, inclusive nas 26 execuções do programa (probe sobre o
recording `tn36-…4543083f`, com a `HudMask` real do agente). Causa: no L2 o bloco que se
move é cor 4/tamanho 14 e o alvo é cor 11/tamanho 14; `multi-align` só pareia objetos de
MESMA cor, e as outras candidatas (`pattern`, `count`, `diversity`) variam só com os
segmentos togglando (ruído de edição). Nenhuma reward candidata mede o par móvel→alvo.

O L1 ensina o par: o objeto que se moveu (tamanho 14) parou em cima de um objeto
**estático de cor 11**. A cor do pouso é estável entre níveis; a cor do móvel não (11 no
L1, 4 no L2 — o programa precisa recolorir). O móvel é aprendível online: o
`MovementModel.observe` já roda em cliques, e no L2 o run move o bloco rigidamente 26×,
logo `avatar_id()` vira o bloco.

Portanto: **não é problema de busca, é reward cega.** Este lever dá gradiente à reward do
nível seguinte a partir dos papéis observados na vitória. Não promete cruzar o L2 do tn36
(exige recolorir + sequência — abordagem "editar-e-confirmar", fora de escopo).

## 2. Escopo

- Entra: módulo `roles.py` (papéis), função `grounded_reach_reward_fn` (goals.py), wiring
  em `agent.py` sob `CAUSAL_REACH`, telemetria, flag nos 2 builders, testes.
- Fora: hill-climb sobre chaves de edição/commit; rotação entre candidatas; mudanças na
  `select_win_reward`; qualquer conhecimento por `game_id`.
- Invariante: com `CAUSAL_REACH=0` o caminho é byte-idêntico. Com `=1`, **nada muda no
  1º nível de nenhum jogo** (papéis só existem após ≥1 level-up) → os 5 níveis do
  baseline cego ficam protegidos por construção.

## 3. Componentes

### 3.1 `agents/causal/roles.py` — papéis aprendidos na vitória

```python
@dataclass(frozen=True)
class WinRoles:
    target_color: int      # cor do objeto estático onde o móvel pousou
    mover_size: int        # tamanho do móvel no frame pré-vitória

def learn_win_roles(pre_objs, win_objs, avatar_obj=None, max_size=100, tol=2) -> WinRoles | None
```

Entradas: objetos (`perception.Object`) da **última cena de decisão** do nível vencido
(`self._prev_scene`) e da **cena de vitória** (`parse(win_grid(...))`); `avatar_obj` =
objeto da cena pré-vitória cujo id é `MovementModel.avatar_id()` (ou `None`).

Algoritmo (numpy/stdlib, exception-safe → `None`):
1. Só objetos com `size <= max_size` (ignora fundo/paredes/HUD).
2. *match(o, objs)*: existe `p` em `objs` com mesma cor, `|size-p.size| <= 2` e centróide
   a distância Manhattan `<= tol`.
3. **Móvel**: se `avatar_obj` dado, é ele. Senão, os objetos pré-vitória SEM match na
   cena de vitória (sumiram da posição). Exige exatamente 1; 0 ou >1 → `None`.
4. **Novos**: objetos da cena de vitória sem match na pré-vitória.
5. **Pouso**: objetos pré-vitória que NÃO são o móvel e cuja bbox intersecta a bbox de
   algum objeto novo. "Estático" aqui = tem match na vitória OU não tem match mas o
   objeto novo que o cobre é maior que ele (fundiu-se com o móvel). Se `avatar_obj`
   dado e a lista vier vazia, usa o estático mais próximo (Manhattan `<= 2`) do match do
   avatar na cena de vitória. 0 candidatos → `None`; >1 → o de tamanho mais próximo de
   `mover_size`.
6. Devolve `WinRoles(target_color=pouso.color, mover_size=móvel.size)`.

Exemplo tn36 L1: móvel = (11,14)@(14,31) sumiu; novos = {(11,30)@(35,31) fundido, ponto
da grade (4,16)@(14,31) reaparecido}; pouso = (11,16)@(35,31) (estático, intersecta o
fundido) → `WinRoles(11, 14)`.

### 3.2 `goals.grounded_reach_reward_fn(mover_color, mover_size, target_color)`

Mesma forma das outras grounded (`state -> (reward, goal_flag)`, exception-safe → `(0.0,
False)`):
- móvel = objeto de `mover_color` com `size` mais próximo de `mover_size`;
- alvo = objeto de `target_color` com `size` mais próximo de `mover_size`, **excluindo o
  próprio móvel** (mesma cor é possível) e objetos com `size > 100`;
- `reward = -manhattan(móvel, alvo)`, `goal_flag = (dist == 0)`; sem móvel ou alvo →
  `(0.0, False)`.

### 3.3 Wiring em `agent.py`

Estado novo (init + reset em `full_reset`): `self._reach_on` (env), `self._win_roles:
WinRoles|None` (persiste entre níveis; a última vitória sobrescreve), `self._flat_steps: int`
(zera no level-up e ao adotar), `self._reach_src: str|None` (telemetria).

1. **Level-up** (bloco onde `select_win_reward` roda): após calcular `win_scene`, se
   `_reach_on`: `avatar_obj` = objeto de `self._prev_scene` com id `avatar_id()` (ou None);
   `self._win_roles = learn_win_roles(prev_scene.objects, win_scene.objects, avatar_obj)`.
2. **Detector de planura** (no fecha-loop decisão→decisão, junto do `_track_rprog`): se
   `_reach_on` e há `_reward_fn` e a transição teve efeito visível (`actual.kind != "none"`
   ou pixel-delta ≥ 5) e `value(depois) == value(antes)` → `_flat_steps += 1`; se o valor
   mudou → `_flat_steps = 0`.
3. **Adoção** (em `choose_action`, antes da pilha decidir, após `_try_learn_reward`): se
   `_reach_on` e `_reach_src is None` e `_win_roles` e `_flat_steps >= REACH_FLAT_K` (=8)
   e `avatar_id()` mapeia a um objeto `av` da cena atual com `av.color` fora das cores de
   fundo → `fn = grounded_reach_reward_fn(av.color, av.size, roles.target_color)`; adota
   só se `accept_reward_fn(fn, self._grounded_states(scene))` passar (rejeita constante /
   falso-positivo; cold-start aceita). Então: `self._reward_fn = fn`,
   `self._reward_src = self._reach_src = f"reach:cor{av.color}#{av.size}->cor{tc}"`,
   `self._win_reward = ("reach", fn, 0.0)` (mantém o `rprog` sem cap decidindo antes do
   `2phase`, caminho 2a já existente), `self._rprog.clear()`, `_flat_steps = 0`.
   Rejeitada → `_reward_rejected += 1`, `_flat_steps = 0` (tenta de novo após mais K).
4. **Precedência** em `_try_learn_reward`: nada muda — como `_win_reward` passa a ser a
   reach, a re-síntese por nível a mantém; um novo level-up recalcula
   `select_win_reward` normalmente (a reach concorre só via adoção, não via rho).
5. **Telemetria** (`phase2_stats`): `win_roles` (`"cor11#14"` ou None), `reach_src`,
   `flat_steps_max`.
6. Flag `CAUSAL_REACH=1` em `kaggle/build_notebook.py` e `build_offline_notebook.py`.

## 4. Testes (TDD)

`tests/causal/test_roles.py`: móvel sumiu + fundiu com estático → papéis certos; avatar
dado sem sumir (pousou visível ao lado) → pouso = estático mais próximo; 2 móveis sem
avatar → None; sem pouso → None; objetos grandes ignorados; entrada ruim → None.
`tests/causal/test_goals.py`: reach exclui o próprio móvel quando cores coincidem; escolhe
alvo por tamanho; sem alvo → (0.0, False); goal em dist 0.
`tests/causal/test_agent_reach.py` (FakeLLM-free, cenas sintéticas): flag off → estado não
existe/caminho inalterado; adota só após K passos com efeito E planos, nunca sem papéis
nem sem avatar; adoção zera `_rprog` e seta `_win_reward=("reach",…)`; `phase2_stats`
traz as chaves; reprovada no `accept_reward_fn` → não adota e conta rejeição.
`tests/kaggle`: flag presente nos 2 ENV; `roles.py` em `MODULES` (o teste-guarda de
imports relativos já pega isso).

## 5. Validação real

1. Probe no recording `tn36-…4543083f`: com papéis do L1 e móvel (4,14), a reward reach
   **varia** nas 26 execuções do L2 (hoje: 1 valor distinto).
2. Loop local cego (`CAUSAL_LLM=0 CAUSAL_GK=0`, 25 jogos, 200 ações): placar `>= 5`;
   tn36 com `reach_src` preenchido e `rprog_fires > 0` no L2.
3. Notebooks regenerados (embed verificado).

## 6. Riscos

- Papéis errados em jogos onde a vitória não é "pousar" (pintura/sequência): o
  `accept_reward_fn` + a exigência de avatar limitam o dano; e a reach só substitui uma
  reward que já estava plana (sem sinal a perder).
- Alvo ambíguo por cor (várias cores 11): desempate por tamanho; se errar, o rprog só
  perde tempo — nunca pior que o round-robin atual.
