# CausalObjectAgent v3 — Descoberta de clique (chave espacial) · Design

> Passo 2 da Fase 5. Fundamentado no run v2 ao vivo (27/ago): cliques do
> `ACTION6` são endereçados no centroide de cada objeto e chaveados por
> `@color,size`, gerando ~77 chaves únicas em 78 ações no vc33 (nunca
> repetem → o modelo causal nunca aprende), enquanto o objeto estável é o
> **fundo** (`color=0,size=848`). Interações reprodutíveis são perdidas:
> clicar em `(62,26)` deu `structural` 3× no vc33, todas sob a mesma chave
> de fundo, sem virar conhecimento explorável.

## Objetivo

Fazer o clique ser **endereçado e chaveado no espaço**, de forma repetível e
learnable, para que o modelo causal aprenda "clicar nesta região faz X" e a
policy possa **varrer** o tabuleiro atrás de células que mexem no jogo e
depois **explorar** as que mexem. Sem LLM/GPU, numpy/stdlib puro,
Kaggle-submittable. Muda apenas o **endereçamento/chave** do clique; a
**medição de efeito** (objeto-cêntrica, HUD-mascarada, IoU) fica intacta.

## Decisões aprovadas

- **Endereçamento = grade espacial (opção A).** Não usar centroides de
  objeto como ponto de clique nem `color,size` como chave.
- **Granularidade = grade 6×6 (36 células)** sobre a grade 64×64.

## Arquitetura

Mudança contida em `agents/causal/policy.py` + testes. `causal_model.py`,
`perception.py`, `hud.py`, `instrumentation.py` e `agent.py` **não mudam** —
o `CausalModel` é agnóstico à string da chave, então basta a policy passar a
nova chave/candidato.

### 1. Grade espacial

Constante de módulo `GRID_N = 6`. A grade 64×64 é particionada em `GRID_N ×
GRID_N` células. Índices de célula `gx, gy ∈ {0..GRID_N-1}` onde `gx` = coluna
(eixo x) e `gy` = linha (eixo y).

- **Ponto de clique = centro da célula:**
  `x = int((gx + 0.5) * 64 / GRID_N)`, `y = int((gy + 0.5) * 64 / GRID_N)`.
  Para `GRID_N=6` isso dá centros em `{5,16,26,37,48,58}` — todos em `0..63`,
  distintos por célula.
- **Célula de um ponto (para presença de objeto):**
  `gx = min(GRID_N-1, x * GRID_N // 64)`, `gy = min(GRID_N-1, y * GRID_N // 64)`.

### 2. Chave de ação

`action_key` passa a ser espacial para ações complexas:

- Ação **simples** (botão): `action.name` — inalterado.
- Ação **complexa** (ACTION6): `f"{action.name}@cell={gx},{gy}"`.

O parâmetro deixa de ser o objeto-alvo e passa a ser a célula. Isso remove a
dependência de `color,size` e garante que cliques na mesma região colapsem na
mesma chave.

### 3. Geração de candidatos

`candidates(scene, available_actions)`:

- Ação simples → 1 candidato `Candidate(action, None, None, action.name, False)`.
- Ação complexa → **um candidato por célula** (`GRID_N*GRID_N` = 36):
  `Candidate(action, x_centro, y_centro, f"{action.name}@cell={gx},{gy}", has_object)`,
  onde `has_object` indica se algum objeto de primeiro plano da `scene`
  (já HUD-mascarada) ocupa aquela célula.

O `Candidate` namedtuple ganha um 5º campo `has_object: bool`.

**Presença de objeto por célula:** a partir de `scene.objects`, mapear cada
objeto para as células que seus `cells` (pixels) tocam; uma célula é
`has_object=True` se ao menos um objeto de primeiro plano a toca. Objetos de
fundo não contam (o fundo é a maior componente / cor de fundo já é ignorada na
segmentação da percepção, então `scene.objects` já exclui o fundo).

### 4. Varredura dirigida (policy.score)

O `score` atual já produz varredura sistemática: célula não-explorada
(`eff is None`) soma **+3.0**; célula sabidamente `none` soma **−2.0** → a
policy esgota células novas antes de repetir uma morta. Acrescentar **um único
termo**: bônus de presença de objeto.

- `if cand.has_object: s += 0.5` — entre células igualmente inexploradas, as
  que contêm objeto são tentadas primeiro (o alvo de interação costuma estar
  sobre/perto de um objeto), sem impedir a cobertura das vazias.

Nenhum outro termo do score muda. Os termos de progresso (`is_progress` → +10)
e de efeito novo (+0.5) continuam valendo e agora operam sobre chaves de célula
estáveis, então uma célula que produz efeito não-`none` reprodutível vira
explorável: `predict(cell_key)` retorna o efeito e o score a mantém viva.

## Fluxo de dados

`agent.choose_action` → `Policy.decide(scene, model, available_actions,
seen_effects, budget_frac)` → `candidates()` lê `scene.objects` (HUD-mascarada)
p/ marcar `has_object` → `score()` escolhe a célula → devolve `Candidate` com
`(x,y)` do centro e chave de célula. O agente já traduz `Candidate.x/y` em
`GameAction.ACTION6.set_data({"x","y"})` e loga a chave — **sem mudança no
agente**. A chave de célula entra em `model.observe(prev, key, curr)` no passo
seguinte (logging deferido já existente).

## Erros e casos de borda

- **Cena sem objetos:** todas as 36 células têm `has_object=False`; a
  varredura ainda cobre o tabuleiro (bônus 0 em todas, empate resolvido pela
  ordem determinística de geração). O fallback antigo "1 candidato no centro"
  deixa de existir — a grade sempre gera 36 candidatos.
- **`GRID_N` que não divide 64:** o cálculo por floats/`//` já trata; centros
  sempre caem em `0..63` e células cobrem toda a grade.
- **Determinismo:** com `epsilon=0`, `decide()` é determinístico (mesma cena +
  mesmo modelo → mesma célula). `epsilon>0` mantém exploração aleatória entre
  os 36 candidatos.

## Testes (TDD, `tests/causal/`)

1. **`action_key` espacial:** complexa na mesma célula → mesma chave;
   células diferentes → chaves diferentes; simples → `action.name`.
2. **cell↔ponto:** centro de cada célula ∈ `0..63`; pontos distintos por
   célula; `point→cell→point` estável.
3. **`candidates()`:** complexa emite 36 candidatos com chaves/pontos
   distintos; simples emite 1; campo `has_object` presente.
4. **presença de objeto:** cena com 1 objeto pequeno numa célula → só aquela
   célula tem `has_object=True`; cena vazia → todas `False`.
5. **prioridade por objeto:** entre células inexploradas, `decide()` escolhe a
   `has_object=True` antes de uma vazia (bônus +0.5).
6. **varredura sem repetição:** após uma célula ser observada `none` no modelo,
   `decide()` prefere uma célula inexplorada (cobertura sistemática).
7. **exploit reprodutível (integração):** clicar duas vezes na mesma região
   gera a mesma chave de célula; um efeito não-`none` registrado nessa chave é
   recuperado por `predict`.
8. **Regressão:** os 50 testes v1+v2 seguem verdes (ações simples e o loop do
   agente não regridem).

## Fora de escopo (próximos passos)

- **Modelo de objetivo / sinal de progresso** (Passo 3) — por que EXPLOIT quase
  nunca dispara. Este design só torna o clique learnable; não ensina o que é
  "vencer".
- Refino coarse-to-fine da grade, chave relacional entre objetos, reuso de
  habilidades entre jogos.

## Critério de pronto

- 36 candidatos por ação complexa, chave de célula estável, `has_object`
  correto; varredura cobre células novas antes de repetir mortas; célula com
  efeito reprodutível é recuperável por `predict`.
- 50 testes v1+v2 + novos testes verdes.
- Rodar ao vivo `vc33` e comparar: nº de chaves de clique distintas deve cair
  de ~77 para ≤36, e chaves com efeito não-`none` devem repetir (accuracy de
  previsão de clique > 0, que no v2 era 0).
