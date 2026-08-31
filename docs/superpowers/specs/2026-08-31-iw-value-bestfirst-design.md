# IW Best-First sobre Reward Denso — Design Spec

> **Data:** 2026-08-31 · **Status:** aprovado · **Escopo:** dar direção ao IW usando o reward escalar do LLM. Default-safe.

## Problema (isolado por telemetria)

O run offline (`sk48`, 32B real) deu o veredito com as 5 chaves de diagnóstico:

```
reward_learned=True · iw_goal_calls=186 · iw_goal_hits=0 · reward_real_true=0/193
reward_src: def reward_function(state):
    if len(state)==1 and state[0][1]['color']==4 and state[0][1]['size']==852:
        return (100.0, True)
    else:
        reward = 100.0 - (len(state)-1)*10.0   # ← scalar denso, HOJE ignorado
        return (reward, False)
```

**Causa (c):** o Qwen sintetiza um predicado de vitória **às cegas** (nunca viu um nível resolvido) com
número mágico hardcoded (`size==852`) exigindo colapso a **1 objeto** exato. Isso é `goal_flag=True`
**nunca** em cena real (0/193) nem alcançável pelo IW/`f_τ` (0/186). O IW goal-directed mira um alvo
impossível → cai na exploração não-dirigida todo passo → 0 níveis.

**Alavanca:** o `reward_function` já devolve um **scalar denso** (`reward`, 1º elemento) que nós
descartamos — só lemos o `goal_flag` binário. Usar o scalar como **heurística de valor** transforma
"nunca atinge a meta exata" em "move rumo a estados de maior reward".

## Objetivo

Fazer o IW procurar, dentro do orçamento de nós, o estado alcançável de **maior reward escalar** e
devolver a **1ª ação** rumo a ele — em vez de exigir `goal_flag=True`. Maximizar valor **subsome** o
goal (um estado-meta tem reward máximo). Mudança mínima, default-safe, testável com FakeLLM.

## Escopo

**Dentro:** `value_fn_from_reward` em `goals.py`; parâmetro `value_fn` em `iw_search`/`iw_plan`
(`iw.py`); `_iw_decide` passa `value_fn` (`agent.py`).

**Fora:** mudar a síntese/validação do reward (isso é a Alavanca A); `goal_fn` de `iw.py` (permanece
p/ não quebrar outros usos); qualquer outro modo da pilha de decisão.

## Componentes

### 1. `goals.py` — `value_fn_from_reward(reward_fn) -> callable`

Irmão do `goal_fn_from_reward` já existente. Adapta `reward_fn(state) → float` lendo o **1º**
elemento (o scalar). À prova de exceção: falha → `float("-inf")` (nunca escolhido).

```python
def value_fn_from_reward(reward_fn):
    """Adapta reward_function -> value(state)->float p/ o IW best-first. Lê o reward
    (1º elemento) e é à prova de exceção (falha → -inf, nunca escolhido)."""
    def value(state):
        try:
            r = reward_fn(state)
            if isinstance(r, (tuple, list)) and len(r) >= 1:
                return float(r[0])
            return float(r)
        except Exception:
            return float("-inf")
    return value
```

### 2. `iw.py` — `value_fn` em `iw_search`/`iw_plan`

`iw_search(start, actions, model, goal_fn=None, value_fn=None, width=1, max_nodes=1000)`.
Precedência: `goal_fn` (se passado) mantém o comportamento atual **inalterado**; senão, se `value_fn`
passado → **modo best-first por valor**; senão → modo exploração width-based atual.

Modo best-first por valor (a poda por novidade de largura — coração do IW — permanece):
- calcula `v0 = value_fn(start)`;
- BFS com poda por novidade (igual hoje), rastreando, entre os estados expandidos, o de **maior
  valor** e a **1ª ação** do caminho até ele (a ação-raiz que originou esse ramo);
- ao fim, devolve essa 1ª ação **só se** o melhor valor for **estritamente maior** que `v0`;
  senão `None` (nada melhora → fallback).

`iw_plan(start, actions, model, goal_fn=None, value_fn=None, max_width=2, max_nodes=1000)` repassa
`value_fn` a cada largura; 1ª que devolve não-`None` vence (igual hoje).

Rastrear a "1ª ação do caminho": a fila guarda `(estado, 1ª_ação)` — já é assim no código atual
(`q.append((nxt, a))` e `q.append((model.predict(state, a), a))`). No modo valor, ao expandir cada
estado, comparo seu `value_fn` e memorizo `(melhor_valor, 1ª_ação)`.

### 3. `agent.py` — `_iw_decide` usa `value_fn`

```python
    def _iw_decide(self, scene, cands):
        if not self._typed.sources:
            return None
        start = [(o.shape_hash, _obj_state(o)) for o in scene.objects]
        vf = value_fn_from_reward(self._reward_fn) if self._reward_fn else None
        r = iw_plan(start, [c.key for c in cands], self._typed, value_fn=vf, max_nodes=300)
        if vf is not None:                       # diag: IW value-directed disparou
            self._iw_goal_calls += 1
            if r is not None:                    # achou ação que melhora o valor
                self._iw_goal_hits += 1
        return r
```

Import: `value_fn_from_reward` junto do `goal_fn_from_reward` já importado (`agent.py:24`).
**Semântica nova dos contadores:** `iw_goal_calls` = IW value-directed disparou (reward viva);
`iw_goal_hits` = achou ação que **melhora** o valor (antes: atingiu meta binária). É a leitura certa
pra este modo e continua diagnosticável.

## Comportamento e segurança

- **Default-safe:** o caminho só muda sob `CAUSAL_IW` on + reward aprendida. Sem reward,
  `value_fn=None` → `iw_plan` cai no modo exploração width-based de hoje (`goal_fn=None,
  value_fn=None`). Outros modos (navigate/plan/η/greedy) intactos.
- **`goal_fn` preservado:** não removo o parâmetro nem o ramo `goal_fn` de `iw.py` → testes e usos
  existentes de `iw_search(goal_fn=...)` continuam verdes.
- **À prova de exceção:** `value_fn_from_reward` engole exceções do código LLM-autorado (→ `-inf`).
- **Sem GPU/LLM aqui:** 100% testável com FakeLLM/reward-lambdas neste ambiente.

## Testes (TDD)

1. `value_fn_from_reward`: lê o scalar `(r,flag)→r`; exceção no reward → `-inf`.
2. `iw_search` best-first: modelo determinístico onde uma ação leva a estado de maior valor →
   retorna essa ação; se nenhuma ação melhora o `start` → `None`.
3. `iw_search` precedência: com `goal_fn` passado, ignora `value_fn` e mantém comportamento atual
   (achado do goal). Sem `goal_fn` e sem `value_fn`, exploração width-based inalterada.
4. `_iw_decide`: com reward densa, `iw_goal_calls` incrementa; `iw_goal_hits` incrementa quando há
   ação que melhora o valor; sem regras aceitas → `None` cedo, contadores não mexem.

## Entregável

`goals.py` + `iw.py` + `agent.py` + testes verdes (320 → +N). Notebook offline reempacota sozinho
(embed dos 3 módulos). Validação real = próximo run offline: observar `iw_goal_hits > 0` e se
`levels_completed` sobe. Se "menos objetos" não for progresso real do jogo, a telemetria mostra e
seguimos p/ a próxima alavanca.
