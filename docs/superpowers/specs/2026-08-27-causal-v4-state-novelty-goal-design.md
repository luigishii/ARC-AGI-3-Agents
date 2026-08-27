# CausalObjectAgent v4 — Modelo de objetivo/progresso (novidade-de-estado) · Design

> Passo 3 da Fase 5. Fundamentado nos runs ao vivo: o agente tem **zero
> level-ups** em qualquer jogo. A policy é curiosidade **action-cêntrica**
> (recompensa ação nova +3, efeito novo +0.5, penaliza `none` −2) e o único
> sinal de progresso (`is_progress`/`progress_keys` → +10) é **retrospectivo**
> — só existe depois de um level-up já ter ocorrido, então nunca dispara e o
> agente nunca direciona comportamento pra vencer; também fixa (ls20 ACTION1
> 57×). Precisamos de um sinal de progresso **intrínseco e cold-start-safe**.

## Objetivo

Dar um sinal de progresso principiado que empurre o agente a **expandir o
conjunto de estados alcançáveis** do jogo (onde os level-ups moram), sem nunca
ter visto um level-up, e mais sério que o "o frame mudou?" ingênuo dos
concorrentes. Trocar o motor de exploração de **novidade-de-ação** para
**novidade-de-ESTADO controlável**. Sem LLM/GPU, numpy/stdlib puro,
Kaggle-submittable. O novo estado é **serializável** (habilita o Passo 4, reuso
entre jogos).

## Decisões aprovadas

- **Sinal = novidade-de-estado (A) + gate de controlabilidade (B).**
- **Assinatura de estado = config de objetos**: por objeto da cena
  HUD-mascarada, `(cor, gx, gy)` com `(gx,gy)` = célula 6×6 do centroide.

## O nó do cold-start (enquadramento honesto)

Com zero level-ups, um "modelo de objetivo" que **persegue uma meta conhecida**
não tem como bootstrapar (a meta é justamente o que nunca vimos). Portanto o
motor de progresso deste passo é a **novidade-de-estado controlável** — a
versão principiada de "progresso rumo a descobrir o objetivo". A perseguição
explícita de uma meta destrava só depois do 1º level-up; para preparar isso, ao
observar um level-up gravamos a **assinatura do estado pré-level-up** como
âncora (apenas registro serializado por ora; o uso na recompensa fica para
depois de haver dados). YAGNI: não construir o mecanismo de perseguição agora.

## Arquitetura

Novo módulo `agents/causal/novelty.py` + mudanças em `agents/causal/policy.py`
(`score`/`decide` ganham parâmetro opcional `novelty=None`) e
`agents/causal/agent.py` (instancia, atualiza e serializa o `NoveltyModel`).
`perception.py`, `hud.py`, `causal_model.py`, `instrumentation.py` **não mudam**.

### 1. Assinatura de estado (`novelty.py`)

`state_signature(scene) -> str`: string JSON-safe e hashable.

```python
def state_signature(scene) -> str:
    parts = []
    for o in scene.objects:
        gx, gy = cell_of(int(round(o.centroid[1])), int(round(o.centroid[0])))
        parts.append((o.color, gx, gy))
    return ";".join(f"{c},{gx},{gy}" for (c, gx, gy) in sorted(parts))
```

Usa `cell_of` de `policy.py` (`x`=col=`centroid[1]`, `y`=row=`centroid[0]`).
Cena vazia → `""`. Objetos de fundo já não entram em `scene.objects`.

### 2. `NoveltyModel` (`novelty.py`, serializável)

```python
OPTIMISTIC_YIELD = 1.0

class NoveltyModel:
    def __init__(self):
        self.counts = {}         # sig(str) -> int (visitas)
        self._yield = {}         # action_key -> [soma_novidade, n]
        self.goal_anchors = []   # list[str] assinaturas pré-level-up

    def count(self, sig): return self.counts.get(sig, 0)
    def novelty(self, sig): return 1.0 / math.sqrt(self.count(sig) + 1)
    def visit(self, sig): self.counts[sig] = self.counts.get(sig, 0) + 1

    def observe_transition(self, key, curr_scene):
        sig = state_signature(curr_scene)
        nov = self.novelty(sig)               # novidade ANTES de contar esta visita
        s, n = self._yield.get(key, [0.0, 0])
        self._yield[key] = [s + nov, n + 1]
        self.visit(sig)

    def yield_estimate(self, key):
        v = self._yield.get(key)
        if not v or v[1] == 0:
            return OPTIMISTIC_YIELD           # init otimista: chave nunca-tentada
        return v[0] / v[1]

    def record_goal_anchor(self, sig):
        if sig not in self.goal_anchors:
            self.goal_anchors.append(sig)

    def to_dict(self): ...   # {counts, yield, goal_anchors}
    @classmethod
    def from_dict(cls, d): ...
```

- `novelty(sig) = 1/√(count+1)` — bônus de contagem de visitação (novidade cai
  quando o estado é revisitado). Estado nunca visto → novidade 1.0.
- `yield_estimate(key)` = média da novidade dos estados que a chave alcançou;
  **otimista** (1.0) para chave sem dados → preserva a exploração cold.

### 3. Gate de controlabilidade + integração no `score` (`policy.py`)

`Policy.score(self, cand, model, seen_effects, budget_frac, novelty=None)` e
`Policy.decide(..., novelty=None)` ganham o parâmetro opcional. Com
`novelty=None` o comportamento é **idêntico ao v3** (mantém os 65 testes).

Com `novelty` presente, o termo de **novidade-de-ação** (`+3 se eff None`;
`+1.5 se conf<0.8`; `+0.5 se efeito novo`) é substituído por um termo de
**novidade-de-estado controlável**:

```python
    def score(self, cand, model, seen_effects, budget_frac, novelty=None):
        eff, conf = model.predict(cand.key)
        s = 0.0
        if model.is_progress(cand.key):
            s += 10.0 * (1 + (1 - budget_frac))
        if novelty is None:                       # caminho v3 (compat)
            if eff is None:
                s += 3.0
            elif conf < 0.8:
                s += 1.5
            if eff is not None and eff.kind not in seen_effects:
                s += 0.5
        else:                                     # caminho v4
            y = novelty.yield_estimate(cand.key)
            ctrl = conf if eff is not None else 1.0   # chave inédita: ctrl otimista
            s += 3.0 * y * ctrl
        if eff is not None and eff.kind == "none":
            s -= 2.0
        if cand.has_object:
            s += 0.5
        return s
```

- Chave inédita: `y=1.0`, `ctrl=1.0` → `+3.0` (igual ao v3 → exploração cold
  preservada).
- Chave que leva a estados novos de forma reprodutível: `y` alto × `conf` alto →
  bônus forte (perseguir).
- Chave que só dá `none`: `−2` e `y` baixo → negativa (abandona célula morta).
- Ruído não-controlável (efeito não-`none` mas `conf` baixo): `ctrl` baixo
  desconta o bônus — **gate B**.

### 4. Wiring no `agent.py`

- `_init_causal_state`: `from .novelty import NoveltyModel, state_signature`;
  `self._novelty = NoveltyModel()`.
- **Não** resetar `self._novelty` no branch de RESET — a memória de visitação
  acumula durante toda a vida do agente (atravessa níveis/tentativas).
- No bloco de fecha-loop (após `actual`/`level_up` calculados):

```python
            if level_up:
                self._novelty.record_goal_anchor(state_signature(self._prev_scene))
            self._novelty.observe_transition(self._last_key, scene)
```

- Passar `self._novelty` para `self._policy.decide(...)`.
- Adicionar `"novelty_yield": round(self._novelty.yield_estimate(cand.key), 3)`
  ao `reasoning` (diagnóstico no log; não muda o schema JSONL — campo extra).

## Fluxo de dados

`choose_action` observa a transição da ação anterior → `observe_transition`
atualiza `yield[key]` com a novidade do estado resultante e conta a visita →
`decide`/`score` usam `yield_estimate(key)·conf` pra pontuar candidatos → a
policy prefere ações que levam a estados novos e controláveis. Level-up →
grava âncora (dado; sem uso na recompensa por ora).

## Erros e casos de borda

- **Cena vazia:** `state_signature` → `""`; conta como um estado como outro
  qualquer (novidade cai se recorrer). Sem exceção.
- **`observe_transition` só é chamado quando há `prev_scene` e `last_key`** (já
  garantido pelo `if` do bloco de fecha-loop).
- **Compat:** `novelty=None` mantém o caminho v3; nenhum teste v1–v3 muda.
- **Serialização:** `counts`, `_yield` (valores `[float,int]`) e `goal_anchors`
  são JSON-safe (chaves string). Roundtrip `to_dict`/`from_dict` testado.
- **Determinismo:** com `epsilon=0`, `decide` é determinístico dado o estado do
  `NoveltyModel`.

## Testes (TDD, `tests/causal/`)

1. **`state_signature`** (`test_novelty.py`): cenas iguais → mesma string;
   objeto em célula diferente → string diferente; cena vazia → `""`; ordem dos
   objetos não importa (ordenado).
2. **`NoveltyModel` contagem/novidade:** `novelty` cai monotonicamente com
   revisitas (`1/√(n+1)`); `visit` incrementa.
3. **`yield_estimate`:** otimista (1.0) sem dados; após `observe_transition`
   com estado novo vs revisitado, reflete a média da novidade.
4. **`observe_transition` + âncora:** atualiza `yield` e `counts`;
   `record_goal_anchor` não duplica.
5. **roundtrip `to_dict`/`from_dict`.**
6. **`score` com novidade** (`test_policy_novelty.py`): chave inédita ainda
   pontua `+3` (paridade com v3); chave com yield alto+conf alto > chave com
   yield alto+conf baixo (gate); chave `none` fica negativa.
7. **`decide` com novidade:** entre candidatos, escolhe o de maior
   novidade-controlável; `novelty=None` reproduz o comportamento v3.
8. **integração no agente** (`test_agent_novelty.py`): após 2 passos,
   `self._novelty` acumulou visita/yield; um level-up simulado grava âncora;
   RESET não zera o `NoveltyModel`.
9. **Regressão:** os 65 testes v1–v3 seguem verdes.

## Fora de escopo (próximos passos)

- **Perseguição explícita de meta** a partir das âncoras (destrava após o 1º
  level-up; precisa de dados).
- **Reuso de habilidades entre jogos** (Passo 4) — o `NoveltyModel`/`CausalModel`
  já nascem serializáveis pra isso.
- Planejamento com forward-model; assinatura de estado mais rica.

## Critério de pronto

- `NoveltyModel` serializável; `score`/`decide` com caminho v4 sob `novelty`;
  agente atualiza/serializa a novidade; RESET não zera.
- 65 testes v1–v3 + novos verdes.
- Rodar ao vivo `vc33`/`ls20` e comparar vs v3: menos fixação (distribuição de
  chaves menos concentrada), `explore` deixando de premiar ação estável-inútil,
  e — meta real — chance de ≥1 level-up. Log em `analysis/out/v4live/`.
