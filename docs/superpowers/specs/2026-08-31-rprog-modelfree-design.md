# Progresso Model-Free por Reward Real (Lever B′) — Design Spec

> **Data:** 2026-08-31 · **Status:** aprovado · **Escopo:** escolher ação pelo delta de reward REAL observado, contornando a cegueira do world-model. Default-safe.

## Problema (provado no 2º run offline)

O Lever B (IW best-first sobre o reward denso) veio **inerte** no `sk48`: stats byte-a-byte iguais
(`iw_goal_hits=0/186`). Causa estrutural (verificada em `typed_model.py:92-103`): `TypedWorldModel.predict`
**preserva a contagem de objetos** (1 saída por objeto de entrada; `f_τ` não cria/destrói) — mas o reward
que o Qwen escreveu depende de `len(state)` (`100-(n-1)*10`). Como `len` nunca muda na simulação, o
`value_fn` é **constante** em todo estado alcançável → paisagem plana → best-first nunca acha melhoria.

**Conclusão:** o modelo `f_τ` não simula a variável que o reward mede. Planejar por ele é cego. Lição
OPINE/Tycho: o gargalo é a qualidade do world-model, não a busca.

## Objetivo

Contornar a limitação do modelo medindo o **efeito REAL** de cada ação no reward denso. Quando uma
transição decisão→decisão ocorre, computar `Δ = value(depois) − value(antes)` e atribuir à ação
executada. Ao decidir, escolher a ação de **maior Δ médio histórico** (se positivo). Hill-climbing
model-free sobre o reward do LLM, usando efeitos observados de verdade.

## Escopo

**Dentro:** estado `self._rprog`/`self._rprog_fires` (`_init_causal_state`); tracker no close-loop;
camada `_rprog_decide` na pilha de decisão; 2 chaves em `phase2_stats`; toggle `CAUSAL_RPROG` no
`.env` dos 2 builders de notebook. Só `agent.py` + os 2 builders + testes.

**Fora:** mudar a síntese do reward; mudar o IW/`f_τ`; qualquer outro modo da pilha.

## Componentes

### 1. Estado (`_init_causal_state`)

```python
self._rprog = {}              # action_key -> [soma_Δ, contagem]
self._rprog_fires = 0         # diag: vezes que a camada rprog escolheu a ação
self._rprog_on = os.environ.get("CAUSAL_RPROG", "0") != "0"
```

### 2. Tracker no close-loop

Junto de `_observe_types` (na transição **decisão→decisão**, mesmo gate que já protege `f_τ`/η das
transições fabricadas de level-up — `agent.py:~152-159`), quando `self._reward_fn is not None`:

```python
if self._reward_fn is not None and self._last_key is not None:
    vf = value_fn_from_reward(self._reward_fn)
    before = [(o.shape_hash, _obj_state(o)) for o in self._prev_scene.objects]
    after = [(o.shape_hash, _obj_state(o)) for o in scene.objects]
    d = vf(after) - vf(before)
    row = self._rprog.setdefault(self._last_key, [0.0, 0])
    row[0] += d
    row[1] += 1
```

`value_fn_from_reward` já é exception-safe (falha → `-inf`); um `-inf` polui o Δ, então o tracker
descarta a amostra se `before`/`after` derem valor não-finito (guarda `math.isfinite`).

### 3. Camada de decisão `_rprog_decide(cands)`

Entre os `cands`, retorna a `key` de **maior Δ médio** (`soma/contagem`) desde que **> 0**; senão
`None`. Sem dados (`contagem==0`) a key não concorre.

```python
def _rprog_decide(self, cands):
    best_key, best_avg = None, 0.0
    for c in cands:
        row = self._rprog.get(c.key)
        if not row or row[1] == 0:
            continue
        avg = row[0] / row[1]
        if avg > best_avg:
            best_avg, best_key = avg, c.key
    if best_key is not None:
        self._rprog_fires += 1
    return best_key
```

Posição na pilha (`agent.py:226-247`): **antes do IW** (inerte) —
`navigate → rprog → IW → plan → η → greedy`. Gate: `if cand is None and self._rprog_on and cands:`.

### 4. Diagnóstico (`phase2_stats`)

```python
"rprog_actions": sum(1 for r in self._rprog.values() if r[1] and r[0] / r[1] > 0),
"rprog_fires": self._rprog_fires,
```

### 5. Toggle no notebook

`build_notebook.py` e `build_offline_notebook.py`: adicionar `"CAUSAL_RPROG=1\n"` ao bloco `.env`
junto de `CAUSAL_ETA`/`CAUSAL_IW`.

## Comportamento e segurança

- **Default-safe:** o tracker só roda com reward aprendida; a camada só sob `CAUSAL_RPROG`. Desligado
  → pilha idêntica de hoje. Sem regressão nos outros modos.
- **Cold-start seguro:** sem dados a camada devolve `None` → fallback explora; ela "esquenta" sozinha.
- **À prova de valor inválido:** descarta amostra com valor não-finito (guarda `math.isfinite`).
- **Sem GPU/LLM aqui:** testável com reward-lambdas/FakeLLM.

## Testes (TDD)

1. tracker: após uma transição decisão→decisão com `reward_fn` que sobe o valor, `self._rprog[key]`
   acumula `Δ > 0`; com reward que abaixa, `Δ < 0`.
2. tracker sem `reward_fn`: `self._rprog` vazio.
3. `_rprog_decide`: escolhe a key de maior Δ médio positivo; incrementa `rprog_fires`; sem key
   positiva → `None`, `rprog_fires` inalterado; sem dados → `None`.
4. `phase2_stats`: `rprog_actions` conta keys com média > 0; `rprog_fires` reflete o contador.
5. builders: o `.env` gerado contém `CAUSAL_RPROG=1`.

## Entregável

`agent.py` + `build_notebook.py`/`build_offline_notebook.py` + testes verdes (331 → +N). **Validação
real = próximo run offline (1 jogo por vez, p/ evitar o gargalo #2):** observar `rprog_fires > 0`
(a camada dirige) e se `levels_completed` sobe. Se dirigir mas não subir nível, a evidência aponta
pro reward (direção errada), não pro mecanismo → próxima alavanca.
