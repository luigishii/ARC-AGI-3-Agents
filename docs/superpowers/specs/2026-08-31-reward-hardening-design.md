# Endurecer a Síntese da Reward — Design Spec

> **Data:** 2026-08-31 · **Status:** aprovado · **Escopo:** aceitação comportamental da reward sintetizada + prompt estrito + self-repair. Default-safe.

## Problema (4 jogos testados offline)

A reward que o Qwen sintetiza é o gargalo comum dos 4 jogos (todos 0 níveis), aceita hoje só por um
check **sintático** (`static_reward_check`: compila + usa `state` + sem estado global). Três modos de
falha passaram:

- **Binária sem gradiente** (vc33/tn36): `return (0, False)` / `(1, True)` → escalar constante → B′
  inerte (`rprog_actions=0`) e IW/value plano.
- **Falso-positivo total** (ls20): `reward_real_true=192/192` — o predicado diz "resolvido" em TODO
  estado real, mas o nível nunca completa.
- **Meta inalcançável / magic-number** (sk48): `len==1 and color==4 and size==852`.

O check sintático não distingue nenhum desses. Precisamos de uma aceitação **comportamental**.

## Objetivo

Avaliar a reward candidata em **estados reais observados** e rejeitar as patológicas (constante /
sempre-True / que quebra), com prompt mais estrito e self-repair. Rejeitar reward ruim **não garante**
reward boa — se o LLM não souber expressar progresso, rejeita tudo → sem reward → exploração pura
(honesto). Diagnóstico mostra o check agindo.

## Escopo

**Dentro:** `accept_reward` em `goals.py`; `_build_reward_prompt` estrito, `_try_learn_reward`
reescrito (amostra do buffer + accept_reward + self-repair), 1 contador em `phase2_stats`
(`reward_rejected`), estado `self._reward_rejected` — só `agents/causal/goals.py` + `agent.py` + testes.

**Fora:** a fixação do B′/greedy (é a recência, lever separado); mudar IW/f_τ; qualquer novo toggle.

## Componentes

### 1. `goals.py` — `accept_reward(source, sample_states, min_states=3) -> (bool, str)`

Irmão comportamental do `static_reward_check`. Retorna `(aceito, motivo)` — o motivo alimenta o repair.

```python
def accept_reward(source, sample_states, min_states=3):
    """Aceitação COMPORTAMENTAL da reward: avalia em estados reais e rejeita patológicas.
    Retorna (aceito, motivo). Cold-start: < min_states estados -> aceita (bootstrap)."""
    fn = compile_reward(source)
    if fn is None:
        return (False, "não compila")
    if len(sample_states) < min_states:
        return (True, "poucos estados p/ julgar (cold-start)")
    vals, flags = [], []
    for st in sample_states:
        try:
            r = fn(st)
        except Exception:
            return (False, "levanta exceção em estado real")
        if isinstance(r, (tuple, list)) and len(r) >= 2:
            vals.append(float(r[0])); flags.append(bool(r[1]))
        else:
            vals.append(float(r)); flags.append(bool(r))
    if all(flags):
        return (False, "goal_flag=True em TODO estado (falso-positivo)")
    distinct_states = len({repr(st) for st in sample_states}) > 1
    if distinct_states and len({round(v, 6) for v in vals}) <= 1:
        return (False, "reward escalar CONSTANTE entre estados distintos (sem gradiente)")
    return (True, "ok")
```

Comportamento por modo de falha:
- **ls20** (sempre `(1,True)`) → `all(flags)` → rejeitado.
- **vc33/tn36** (sempre `(0,False)` em estados reais distintos) → escalar constante → rejeitado.
- **sk48** (`100-(n-1)*10` varia com n) → gradiente real → **aceito** (fixação é outro lever).
- reward que **levanta** em estado real → rejeitado.
- `distinct_states=False` (agente não moveu) → pula o teste de gradiente (não dá pra julgar),
  mantém só o teste de falso-positivo → evita rejeitar reward boa por falta de variação.

### 2. `agent.py` — `_build_reward_prompt` estrito

Acrescentar ao prompt: (a) **reward GRADUADA** — número maior = mais perto de resolver, **não** só
0/1; (b) **não** hardcodar tamanhos/posições exatos (magic numbers); (c) `goal_flag=True` **só** quando
o nível está realmente resolvido (raro).

### 3. `agent.py` — `_try_learn_reward` reescrito

- Monta `sample_states` = estados dos cenários do `self._buffer` + a cena atual:
  `[[(o.shape_hash, _obj_state(o)) for o in sc.objects] for (sc, _, _) in self._buffer] + [cena_atual]`.
- Para cada candidata (das N amostras do LLM): `static_reward_check(src)` **e**
  `accept_reward(src, sample_states)[0]`. Aceita a 1ª que passa ambos.
- Se nenhuma passa e `_repair_max > 0`: self-repair — re-pergunta com o `motivo` da rejeição
  (via `_build_reward_repair_prompt`), até `_repair_max` vezes (reusa o padrão do f_τ).
- Toda rejeição (comportamental) incrementa `self._reward_rejected`.
- Se nada passa após repair → `_reward_fn = None` (= síntese falha de hoje; cai na exploração).

### 4. `phase2_stats` — diagnóstico

```python
"reward_rejected": self._reward_rejected,
```

`self._reward_rejected = 0` init em `_init_causal_state`.

## Comportamento e segurança

- **Default-safe:** aperto estrito no caminho de síntese que já roda sob `CAUSAL_LLM`/`CAUSAL_TYPED`;
  sem novo toggle. Cold-start (`min_states`) preserva bootstrap e testes que não populam o buffer.
- **Rejeitar → None** é aceitável: reward constante já deixava B′ inerte; falso-positivo já era inútil.
- **À prova de exceção:** `accept_reward` engole exceções da reward LLM-autorada (→ rejeita).
- **Sem GPU/LLM aqui:** 100% testável com FakeLLM + estados sintéticos.

## Testes (TDD)

1. `accept_reward`: sempre-True → rejeita (falso-positivo); escalar constante em estados distintos →
   rejeita (sem gradiente); graduada que varia → aceita; que levanta → rejeita; `< min_states` →
   aceita (cold-start); estados idênticos + escalar constante → aceita (pula gradiente).
2. `_try_learn_reward`: com `FakeLLM` devolvendo reward binária + buffer de estados distintos →
   rejeita, `reward_rejected` incrementa, `_reward_fn` fica None; com reward graduada → aceita.
3. self-repair: 1ª resposta binária (rejeitada) → 2ª graduada (aceita) sob `CAUSAL_REPAIR>=1`.
4. `_build_reward_prompt`: contém as instruções de graduada / sem-magic / goal_flag-raro.
5. `phase2_stats`: expõe `reward_rejected`.

## Entregável

`goals.py` + `agent.py` + testes verdes (baseline atual + N). Notebook offline reembala `agent.py`/
`goals.py` sozinho. **Validação real = próximo run offline (multi-jogo):** observar `reward_rejected > 0`
nos jogos de reward ruim e se algum passa a ter reward graduada/válida (→ B′/IW ganham sinal). Se o
LLM não conseguir reward boa, rejeita tudo (honesto) → próximo lever = Tycho (modelo opcional).
