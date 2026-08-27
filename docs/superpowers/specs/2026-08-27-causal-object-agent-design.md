# Design — CausalObjectAgent (aposta central v1, ARC-AGI-3)

> Data: 2026-08-27 · Fase 3 (brainstorm da aposta central) · Projeto ARC Prize 2026 / ARC-AGI-3
> Status: **design aprovado, pronto para plano de implementação**

## 1. Contexto e motivação

Competição: agente interativo para o ARC-AGI-3 (Kaggle). Observação = grade 64×64 de
inteiros 0–15; ações RESET + ACTION1–7 (algumas complexas exigem `x,y` em 0–63);
orçamento ~80 ações/jogo; score = níveis completados (eficiência conta); eval privada
usa **jogos não vistos**; sem internet e open-source obrigatório na avaliação.

A análise de 6 notebooks competidores (ver `CLAUDE.md` → "Inteligência competitiva")
confirmou três brechas exploráveis — **ninguém as ataca bem**:

1. **Sinal de progresso universal ingênuo:** todos usam "o frame mudou?" como reward
   proxy (recompensa ruído/animação, sem direção principiada).
2. **Zero reuso de aprendizado:** goose e "Forge" reinstanciam a rede a cada nível;
   ninguém transfere habilidades entre níveis/jogos.
3. **Representação pobre:** pixels crus (CNN) ou ASCII+segmentação para LLM; pouco
   relacional/objeto-cêntrico de verdade.

Melhor score de milestone observado nos concorrentes ≈ **1.21 níveis**. Teto de
novidade conceitual 2–3/5. O espaço para uma aposta principiada está aberto.

## 2. A aposta

Um agente que **não "joga", mas constrói um modelo causal interpretável do jogo**
enquanto age, e decide a partir dele. Tudo em **numpy puro, sem GPU, sem LLM**,
cabendo no orçamento de ~80 ações.

A aposta empacota as três brechas, mas o **núcleo do v1** é a fundação principiada:

- **Núcleo v1:** representação **objeto-cêntrica** + **modelo causal simbólico**
  (`ação → efeito no objeto`).
- **Semente barata no v1:** exploração por **ganho de informação** embutida como um
  termo de score na Policy (substitui o "frame mudou").
- **Diferido (Fase 2+):** reuso de habilidades entre jogos (o modelo já nasce
  serializável para habilitar isso sem refatorar).

**Alinhamento à filosofia ARC (Chollet/Knoop):** interpretável, eficiente em amostras,
sem compute gigante, objetivo principiado — exatamente os eixos premiados e o oposto
do que a maioria dos concorrentes faz.

## 3. Arquitetura

`CausalObjectAgent(Agent)` — subclasse do `Agent` base do harness
(`agents/agent.py`); implementa `choose_action(frames, latest_frame) -> GameAction`
e `is_done(...)`. Orquestra quatro componentes isolados e testáveis:

| Componente | Responsabilidade | Interface (entrada → saída) |
|---|---|---|
| **`Perception`** | Segmenta a grade 64×64 em objetos (connected-components por cor) e casa identidades entre frames. | `FrameData` → `Scene` (objetos + índice espacial) |
| **`CausalModel`** | Tabela de transições observadas; acumula efeito por `(ação[,alvo])`; prevê efeitos. | `(Scene_t, ação, Scene_t+1)` → atualiza; `predict(ação, Scene)` |
| **`Policy`** | Escolhe a ação por score interpretável; máquina de estados EXPLORAR/EXPLOITAR. | `Scene, CausalModel, available_actions` → `GameAction` |
| **`Instrumentation`** | Loga ação (id+x,y+reasoning) e métricas do modelo em JSONL. | eventos → JSONL |

### Fluxo por passo (`choose_action`)

```
FrameData → Perception.parse() → Scene_t
         → [se houve ação anterior] CausalModel.observe(Scene_{t-1}, last_action, Scene_t)
         → Policy.decide(Scene_t, CausalModel, available_actions) → action
         → Instrumentation.log(action, model_stats)
         → guarda (Scene_t, action) para o próximo observe
```

## 4. Componentes em detalhe

### 4.1 `Perception` (perception.py)

- **Objetos:** connected-components por cor (4/8-vizinhança sobre a grade 64×64).
- **Atributos por objeto:** `id, cor, bbox, centroide, tamanho, forma-hash`.
- **Identidade persistente:** matching de objetos entre frames consecutivos por
  (forma-hash + cor + proximidade de centroide), dando `id` estável ao longo do jogo.
- **Saída:** `Scene` = lista de `Object` + índice espacial para consulta rápida por
  posição (usado pela Policy para ancorar cliques).

### 4.2 `CausalModel` (causal_model.py)

- **Efeito** entre `Scene_t` e `Scene_{t+1}`, categorizado: `objeto moveu Δ(dx,dy)`,
  `apareceu`, `sumiu`, `mudou cor`, `nada`, `HUD/score mudou`.
- **Regra:** chave `(ação, contexto-do-alvo)` onde contexto = tipo/cor/tamanho do
  objeto sob o clique (para ações complexas) ou global (para ações simples);
  valor = distribuição de efeitos observados com contagem e confiança.
- **API:** `observe(scene_prev, action, scene_next)` atualiza; `predict(action, scene)`
  retorna efeito esperado + confiança; `stats()` expõe cobertura e estabilidade.
- **Serializável:** `to_dict()`/`from_dict()` (JSON) — não usado no v1, gancho da Fase 2.

### 4.3 `Policy` (policy.py) — o coração

**Candidatos a cada passo:**
- Ações simples disponíveis (`ACTION1–5`): 1 candidato cada.
- Ações complexas disponíveis (`ACTION6/7`): **não varre 4096 coords**; candidatos =
  **centroides dos objetos** da Scene + poucos pontos salientes (cantos de bbox).
  Ancora o clique em objetos, não em pixels aleatórios.

**Score interpretável** (soma ponderada, pesos fixos e legíveis):

| Termo | Sinal | Peso | Racional |
|---|---|---|---|
| Progresso conhecido | regra já leva a `levels_completed++` ou efeito encadeável rumo a isso | alto (exploit) | se sabemos vencer, vencemos |
| Ganho de informação | quão desconhecido/incerto é o efeito de `(ação, alvo)` (nunca tentado > 1× > instável) | médio (explore) | semente principiada; substitui "frame mudou" |
| Novidade de estado | Scene prevista é inédita vs. Scenes já vistas (hash) | baixo | evita loops improdutivos |
| Penalidade de estagnação | efeito conhecido = "nada" | negativo | não desperdiça ações |

**Decisão:** `argmax(score)`; empate → menor custo de exploração já gasto no par;
ε pequeno de aleatoriedade só para sair de platôs.

**Máquina de estados (2 modos):**
1. **EXPLORAR** (padrão inicial): domina o ganho-de-informação; testa cada ação
   simples ≥1× e sonda objetos com as complexas; preenche a tabela causal rápido.
2. **EXPLOITAR** (quando surge regra rumo ao objetivo): segue/encadeia a regra. Se o
   efeito previsto ≠ observado (modelo furado), volta a EXPLORAR e corrige a regra.

**Objetivo sem instruções:** `levels_completed++` (da `FrameData`) é a **única
recompensa terminal dura**. Mudanças estruturais grandes na Scene são **proxy fraco só
de diagnóstico**, nunca reward primário (lição do "graft" do duck: diagnóstico medido,
não predição que envenena a decisão).

**Orçamento:** a Policy conhece `action_counter`/`MAX_ACTIONS`; conforme o budget
encolhe, desloca o peso EXPLORAR→EXPLOITAR.

### 4.4 `Instrumentation` (instrumentation.py)

Loga por passo em JSONL: `id, x, y, reasoning, modo, efeito_previsto,
efeito_observado, model_stats`. Consumível pelo `analysis/replay.py` existente
(estendido para plotar curvas de acurácia/cobertura). Resolve o gotcha do harness
(o `action_input` das gravações oficiais é placeholder — logamos a ação nós mesmos).

## 5. Validação e critério de sucesso

**Métricas-alvo (movem o leaderboard):**
- `levels_completed` por jogo e **ações-por-nível**.
- Jogos com ≥1 nível vs. baseline random/heurístico.

**Métricas de diagnóstico do modelo causal:**
- **Acurácia de previsão 1-passo:** % de acerto do efeito previsto vs. observado.
- **Cobertura:** fração do espaço `(ação × tipo-de-objeto)` observada ≥1×.
- **Estabilidade de regra:** % de regras determinísticas vs. instáveis.
- **Perfil de decisão:** ações em EXPLORAR vs. EXPLOITAR; ações desperdiçadas ("nada").

**Loop de teste (dev):**
1. **Unit tests isolados** (sem API) por componente: `Perception` (grades sintéticas →
   objetos esperados), `CausalModel` (sequências → regras esperadas), `Policy`
   (modelo+scene mockados → ação esperada). Permite iterar sem gastar orçamento de API.
2. **Integração local:** `uv run main.py --agent=causalobject --game=ls20` (+2–3 jogos
   públicos); comparar métricas vs. random.
3. **Gate de "v1 pronto":** completar ≥1 nível num jogo que o random não completa **E**
   acurácia de previsão >70% em jogos com ≥20 passos explorados.

## 6. Estrutura de arquivos

```
agents/causal/
  __init__.py
  perception.py      # Scene, Object, parse(), match_objects()
  causal_model.py    # CausalModel: observe(), predict(), stats(), to_dict/from_dict
  policy.py          # decide(), scoring, FSM EXPLORAR/EXPLOITAR
  instrumentation.py # JSONL logger + métricas do modelo
  agent.py           # CausalObjectAgent(Agent) — orquestra os 4
tests/causal/        # unit tests por componente
```

Registrar o agente em `agents/__init__.py` como `"causalobject"`.

## 7. Fora de escopo do v1 (YAGNI)

Reuso de habilidades entre jogos; prior Bayesiano formal; encadeamento de habilidades
multi-passo; qualquer LLM; qualquer GPU. São camadas **posteriores**, habilitadas pelo
núcleo (o `CausalModel` serializável é o único gancho que já deixamos pronto).

## 8. Riscos e mitigações

- **Segmentação por cor pode não casar a noção de "objeto" do jogo** (ex.: objetos
  multicoloridos). Mitigação: forma-hash + matching tolerante; iterar via unit tests
  com grades sintéticas antes de gastar API.
- **Modelo causal instável se o jogo tem dinâmica estocástica/temporal.** Mitigação: a
  categoria de efeito inclui "instável"; a Policy trata regra instável como baixa
  confiança e volta a explorar.
- **80 ações podem ser poucas para explorar + explorar.** Mitigação: candidatos
  complexos restritos a centroides (não 4096 coords); shift de budget EXPLORAR→EXPLOITAR.
- **Over-engineering** (o erro recorrente dos concorrentes). Mitigação: escopo v1
  enxuto, 4 componentes pequenos, gate de "pronto" objetivo, YAGNI explícito.
