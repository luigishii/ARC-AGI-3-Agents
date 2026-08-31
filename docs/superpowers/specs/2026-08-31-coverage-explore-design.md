# Exploração por Cobertura + Anti-Fixação — Design Spec

> **Data:** 2026-08-31 · **Status:** aprovado · **Escopo:** camada de exploração que varre o espaço de ações em vez de fixar. Default-safe.

## Problema (diagnosticado no vc33 + 4 jogos)

Análise do recording do vc33 (nosso log `analysis/out/causal_vc33.jsonl` cruzado com os frames):
o agente clica **sempre no mesmo objeto** — a faixa do topo (`ACTION6@color=7,size=64→63→...`) —
consumindo o orçamento sem sondar os outros objetos (avatar amarelo, alvo vermelho, roxo, abertura
preta). O `size` muda a cada clique → chave nova toda vez → **nunca aprende** → fica em `EXPLORE`
no mesmo lugar. Mesma **FIXAÇÃO** vista em sk48 (ACTION3 ~185×) e tn36/vc33 (ACTION6 200×).

O nível 0 do vc33 resolve em **7 ações** (baseline); o agente faz 200+ **na mesma coisa** e não
cruza. O gargalo não é reward/navegação/IW — é o **fallback guloso que fixa e não cobre o espaço**.

## Objetivo

Trocar/anteceder o fallback guloso por uma camada que escolhe sempre a candidata **menos visitada**
→ nenhuma ação repete até todas terem sido tentadas. Quebra a fixação e cobre as 36 células do grid
+ ações — pré-requisito pra achar a sequência que resolve. Game-agnóstico (clique e teclado).

## Escopo

**Dentro:** estado `self._cover` (contagem por key) + `self._cover_on`; contagem no close-loop;
camada `_cover_decide` na pilha (entre η e greedy); 1 chave em `phase2_stats` (`cover_keys`);
toggle `CAUSAL_COVER` nos 2 builders. Só `agents/causal/agent.py` + os 2 builders + testes.

**Fora:** encadear ações em sequência ordenada (forward-model sobre ações com-efeito — follow-up);
skip de células mortas/no-op (follow-up); mudar reward/IW/navigate.

## Componentes

### 1. Estado (`_init_causal_state`)

```python
self._cover = {}              # action_key -> nº de vezes que a ação foi tomada
self._cover_on = os.environ.get("CAUSAL_COVER", "0") != "0"
```

### 2. Contagem no close-loop

Na transição decisão→decisão (mesmo bloco de `_observe_types`/`_track_rprog`), conta a ação tomada:

```python
if self._last_key is not None:
    self._cover[self._last_key] = self._cover.get(self._last_key, 0) + 1
```

### 3. Camada `_cover_decide(cands)`

Escolhe a candidata de **menor contagem** de cobertura. Desempate, em ordem:
(a) **`has_object`** True antes de False (célula com objeto > vazia);
(b) evita **`self._last_key`** (anti-repetição imediata);
(c) determinístico (ordem de `cands`).

```python
def _cover_decide(self, cands):
    def rank(c):
        return (self._cover.get(c.key, 0),      # menos visitada primeiro
                0 if c.has_object else 1,        # objeto antes de vazio
                1 if c.key == self._last_key else 0)  # evita repetir a última
    best = min(cands, key=rank)
    return best.key
```

Posição na pilha (`agent.py:238-262`): **entre η e greedy** —
`navigate → rprog → IW → plan → η → COVER → greedy`. Gate: `if cand is None and self._cover_on and cands:`.

Efeito: **sweep sistemático** — cada célula/ação é tentada uma vez antes de qualquer repetição. No
vc33, em vez de martelar a faixa, varre as 36 células (incluindo avatar/vermelho/roxo/abertura).

### 4. Diagnóstico (`phase2_stats`)

```python
"cover_keys": len(self._cover),
```

Fixação → poucas keys distintas; sweep → muitas.

### 5. Toggle no notebook

`build_notebook.py` e `build_offline_notebook.py`: adicionar `"CAUSAL_COVER=1\n"` ao `.env` junto de
`CAUSAL_RPROG`.

## Comportamento e segurança

- **Default-safe:** camada só sob `CAUSAL_COVER` (default off) → pilha idêntica sem o toggle; o
  fallback greedy e seus testes ficam intactos.
- **Cold-start seguro:** `self._cover` vazio → todas as candidatas empatam em 0 → desempate
  has_object/ordem escolhe uma; a partir daí varre.
- **Game-agnóstico:** ranqueia ações simples E células de clique pela mesma contagem.
- **Sem GPU/LLM aqui:** testável com Candidates sintéticos.

## Testes (TDD)

1. `_cover_decide` varre: com `self._cover` vazio e 3 candidatas → escolhe uma; após marcar essa
   como visitada (`self._cover[key]=1`), a próxima escolha é **outra** (menos visitada).
2. desempate `has_object`: entre duas de contagem 0, escolhe a com `has_object=True`.
3. anti-repetição: entre duas de contagem 0 e mesmo `has_object`, evita `self._last_key`.
4. contagem no close-loop: após uma transição decisão→decisão, `self._cover[last_key]` incrementa.
5. `phase2_stats`: expõe `cover_keys` = nº de keys distintas cobertas.
6. builders: o `.env` gerado contém `CAUSAL_COVER=1`.

## Entregável

`agent.py` + os 2 builders + testes verdes (baseline atual + N). Notebook offline reembala `agent.py`
sozinho. **Validação real (offline, foco vc33 via `OFFLINE_GAMES="vc33"`):** observar `cover_keys`
subir muito (varreu as células, não fixou) e se `levels_completed` sobe. Se cobrir mas não cruzar →
a solução exige **sequência ordenada** → próximo lever = encadear via forward-model sobre as ações
que provaram ter efeito. Trava anti-overfit: validar depois nos outros 3 jogos (mecanismo é geral).
