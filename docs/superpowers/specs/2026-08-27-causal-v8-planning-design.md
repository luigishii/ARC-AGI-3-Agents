# CausalObjectAgent v8 — Planejamento com forward-model · Design

> Passo 5 da Fase 5. Teste decisivo: vc33 com 2000 ações + prior deu **0
> níveis** — o gargalo não é escala, é que o agente **explora mas não persegue
> meta**. Escolha do usuário: **planejamento (opção A)** com objetivo de
> **fronteira/novidade** quando não há âncora.

## O que falta tecnicamente

O `CausalModel` prevê só o **tipo de efeito** por chave (moved/structural/none),
não o **próximo estado**. Para planejar/simular precisamos de um **modelo de
transição de estado**: `(state_sig, action_key) → distribuição de próxima
state_sig`, aprendível das transições que já observamos no fecha-loop do agente
(`state_signature(prev) , key , state_signature(curr)`).

## Decisões aprovadas

- **Planejamento (A)** com **beam curto** sobre o modelo de transição.
- **Objetivo sem âncora = fronteira/novidade** (estado terminal menos
  visitado); **com âncora** (pós-1º-level-up) = distância à âncora (goal-directed).
- **1ª versão mínima e testável.**

## Arquitetura

Novo módulo `agents/causal/planning.py` (`TransitionModel` + `plan`) + wiring no
`agents/causal/agent.py` (híbrido: planeja se conseguir, senão cai na policy
gulosa). `perception/hud/causal_model/policy/novelty/transfer/instrumentation`
**não mudam** (o planner consome `state_signature`, `NoveltyModel`,
`cell_center` já existentes).

### 1. `TransitionModel` (`planning.py`, serializável)

```python
class TransitionModel:
    def __init__(self):
        self.trans = {}   # sig(str) -> {action_key -> {next_sig -> count}}

    def observe(self, prev_sig, key, next_sig):
        d = self.trans.setdefault(prev_sig, {}).setdefault(key, {})
        d[next_sig] = d.get(next_sig, 0) + 1

    def predict_next(self, sig, key):
        d = self.trans.get(sig, {}).get(key)
        if not d:
            return None                      # (sig,key) inédito → estado desconhecido
        return max(d.items(), key=lambda kv: kv[1])[0]

    def known_keys(self, sig):
        return list(self.trans.get(sig, {}).keys())

    def to_dict(self) / from_dict(cls, d)     # dicts aninhados str→str→str→int (JSON-safe)
```

Tudo com chaves string (sigs e keys) → JSON-safe, serializável (encaixa no
padrão v1-v6; futura persistência via 4b se quisermos).

### 2. Planner (`plan`)

`plan(start_sig, start_keys, tmodel, novelty, anchors, depth=3, beam=8) -> str | None`

- Busca em **beam** (largura `beam`, profundidade `depth`) sobre sequências de
  `action_key`. No nó inicial ramifica pelas `start_keys` (chaves dos candidatos
  reais da cena atual); em nós futuros ramifica por `tmodel.known_keys(sig)`.
- Transição simulada = `tmodel.predict_next(sig, key)`. Se `None` (par inédito),
  o ramo termina num **estado-fronteira** (score máximo — é justamente o que
  queremos explorar).
- **Score do estado terminal:**
  - **com âncora** (`anchors` não-vazio): `-min_a dist(sig, a)`, `dist` = nº de
    tuplas `(cor,gx,gy)` que diferem (assinaturas são `";"`-joined → comparar
    conjuntos). Menor distância = melhor.
  - **sem âncora:** novidade `novelty.novelty(sig)`; estado-fronteira
    (`predict_next None`) → `1.0` (máx).
- Retorna a **1ª ação** do melhor plano completo, ou `None` se nenhum plano
  supera "não sei" (ex.: `start_sig` sem nenhuma chave conhecida e sem
  candidatos — cai no fallback).

Custo: `beam × depth × ramificação` — barato (beam 8, depth 3).

### 3. Wiring no `agent.py` (híbrido plan-or-fallback)

- `_init_causal_state`: `self._tmodel = TransitionModel()`; `self._last_sig = None`.
- No fecha-loop (após `actual`), registrar a transição:
  ```python
  cur_sig = state_signature(scene)
  if self._last_sig is not None and self._last_key is not None:
      self._tmodel.observe(self._last_sig, self._last_key, cur_sig)
  ```
- Ao decidir: montar os candidatos (`candidates(scene, available)`), tentar
  `plan(cur_sig, [c.key for c in cands], self._tmodel, self._novelty,
  self._novelty.goal_anchors)`. Se retornar uma chave, escolher o candidato com
  aquela chave (reconstruindo `x,y` do candidato — já vem no `Candidate`). Senão,
  `self._policy.decide(...)` (comportamento v7).
- Guardar `self._last_sig = cur_sig` junto de `self._last_key`.
- **Toggle** `CAUSAL_PLAN` (env): default ligado; `CAUSAL_PLAN=0` desliga (A/B
  contra o v7 guloso).

## Fluxo de dados

Transição anterior → `TransitionModel.observe` aprende `(sig,key)→next_sig` →
ao decidir, o planner simula sequências e mira o estado terminal mais novo
(ou a âncora, se houver) → devolve a 1ª ação; se não sabe o bastante, cai na
policy gulosa (que constrói mais transições). Ciclo: quanto mais joga, mais o
modelo enche e o planning aprofunda.

## Erros e casos de borda

- **Modelo vazio / estado inédito:** `plan` devolve `None` → fallback guloso
  (constrói o modelo). Cold-start seguro.
- **Ramificação explosiva:** limitada por `beam`/`depth` fixos e por
  `known_keys` (só chaves já observadas em nós futuros).
- **Chave planejada sem candidato correspondente** (ex.: cena mudou e a chave
  não está entre os candidatos atuais): fallback guloso.
- **Determinismo:** dado o modelo, `plan` é determinístico (desempate por ordem).
- **Compat:** `CAUSAL_PLAN=0` reproduz exatamente o v7; nenhum teste v1–v7 muda.

## Testes (TDD, `tests/causal/`)

1. **`TransitionModel`** (`test_planning.py`): `observe`/`predict_next` (modal),
   `known_keys`, par inédito → `None`; roundtrip `to_dict`/`from_dict`.
2. **`plan` alcança estado novo (MDP de brinquedo):** com um `TransitionModel`
   semeado (A→s1 via k1, s1→s2 via k2, ...), o planner prefere a chave que leva
   ao estado terminal de maior novidade; par inédito (fronteira) é preferido a um
   estado já muito visitado.
3. **`plan` goal-directed com âncora:** dado um anchor, o planner prefere a chave
   cujo plano chega mais perto (menor `dist`).
4. **`plan` sem dados → `None`** (fallback).
5. **integração no agente** (`test_agent_planning.py`): após passos, o
   `TransitionModel` acumulou transições; com `CAUSAL_PLAN=0` o agente reproduz o
   caminho v7 (não usa plano); ligado, usa o plano quando há.
6. **Regressão:** os 117 testes v1–v7 seguem verdes.

## Fora de escopo

- Persistência do `TransitionModel` em disco (pode reusar o padrão 4b depois).
- Objetivos mais sofisticados (empowerment); modelo de transição probabilístico
  (usamos o modal).

## Critério de pronto

- `TransitionModel` aprende e serializa; `plan` mira novidade/fronteira (ou
  âncora) via beam; agente é híbrido plan-or-fallback com toggle `CAUSAL_PLAN`.
- 117 testes v1–v7 + novos verdes.
- **Teste ao vivo (a pergunta real):** rodar 1+ jogo com orçamento alto
  (`CAUSAL_MAX_ACTIONS≈2000`) com planning ON e comparar vs OFF — cruza ≥1
  nível? Se não, evidência de que o approach A não basta (→ B/C).
