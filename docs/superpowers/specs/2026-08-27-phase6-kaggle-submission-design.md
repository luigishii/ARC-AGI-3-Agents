# Fase 6 — Notebook de submissão Kaggle + MAX_ACTIONS por env · Design

> Empacotar o agente `causalobject` (v1–v6, 109 testes verdes) num notebook
> Kaggle público (open-source obrigatório), shipando o `TransferPrior`
> pré-treinado read-only. Fecha o caminho até o leaderboard.

## Contexto técnico verificado

- O pacote `agents/causal/` importa apenas **numpy + arcengine (wheel) +
  stdlib** — offline-safe (sem matplotlib/langgraph/smolagents).
- 9 módulos a levar: `__init__.py`, `agent.py`, `causal_model.py`, `hud.py`,
  `instrumentation.py`, `novelty.py`, `perception.py`, `policy.py`,
  `transfer.py` + o artefato `prior.json`.
- O repo shipado no Kaggle é o da **competição** (sem nosso código) → o
  notebook precisa injetar nosso pacote no ambiente.
- `MAX_ACTIONS = 80` é hardcoded; na eval (wall-clock, cap alto/inf) isso nos
  limitaria — deve virar configurável por env.

## Decisões aprovadas

- **Entrega = notebook auto-contido gerado por build-script** (sem dataset
  externo; código embutido e visível → open-source-friendly; fonte única = o
  repo).
- **Pré-treino = todos os jogos** (swarm ao vivo com `CAUSAL_PRIOR_SAVE=1` →
  `agents/causal/prior.json` commitado).
- **`MAX_ACTIONS` configurável por env** (default 80 local; alto no notebook).

## Arquitetura

Três peças:

1. **`prior.json`** (6a) — artefato de dados gerado pelo pré-treino ao vivo
   (fora do código; commitado em `agents/causal/prior.json`).
2. **`MAX_ACTIONS` por env** — mudança pontual em `agents/causal/agent.py`.
3. **`kaggle/build_notebook.py`** — build-script que lê o pacote + prior e emite
   `kaggle/submission.ipynb`.

`policy/perception/hud/causal_model/novelty/transfer/instrumentation` **não
mudam**.

### 1. `MAX_ACTIONS` por env (`agent.py`)

Em `_init_causal_state`, definir o cap por instância a partir da env, com
fallback no default de classe:

```python
        self.MAX_ACTIONS = int(os.environ.get("CAUSAL_MAX_ACTIONS", type(self).MAX_ACTIONS))
```

- Classe mantém `MAX_ACTIONS = 80` (default local/testes).
- Lido na **instanciação** (após `load_dotenv`), então o notebook seta
  `CAUSAL_MAX_ACTIONS` no `.env` e o valor vale na eval.
- O `budget_frac` (que usa `self.MAX_ACTIONS`) continua funcionando; com cap
  alto, `budget_frac ≈ 1` a maior parte do jogo (exploração sustentada).

### 2. Build-script (`kaggle/build_notebook.py`)

Lê cada módulo de `agents/causal/*.py` e o `prior.json`, **base64-encoda**
(à prova de colisão de aspas/backticks), e monta um notebook nbformat v4
(`{"cells": [...], "metadata": {...}, "nbformat": 4, "nbformat_minor": 5}`)
gravado em `kaggle/submission.ipynb`. Funções puras + `main()`:

- `read_sources(root) -> dict[str, str]`: `{caminho_relativo: base64}` para os 9
  módulos + `prior.json` (se existir; ausente → não embute, notebook não escreve
  prior e a eval roda sem warm-start).
- `build_notebook(sources) -> dict`: monta o dict do notebook.
- `main()`: `read_sources` → `build_notebook` → grava `kaggle/submission.ipynb`.

**Células do notebook gerado:**

- **Cell 0 (code):** instala wheels offline
  `!pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv`.
- **Cell 1 (code):** ramo rerun. Em Python:
  - `if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):`
  - espera o gateway (`!curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games`);
  - copia o repo da competição pra `/kaggle/working/ARC-AGI-3-Agents`;
  - `FILES = { "<rel>": "<b64>", ... }` (embutido) → loop que `base64.b64decode`
    e grava cada arquivo em `/kaggle/working/ARC-AGI-3-Agents/<rel>`
    (`os.makedirs` do dir pai);
  - sobrescreve `agents/__init__.py` enxuto:
    ```python
    from typing import Type
    from dotenv import load_dotenv
    load_dotenv()                      # ANTES de importar o agente (env do .env)
    from .agent import Agent, Playback
    from .swarm import Swarm
    from .causal.agent import CausalObjectAgent
    AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"causalobject": CausalObjectAgent}
    ```
  - escreve `.env` roteando pro gateway + prior read-only:
    ```
    SCHEME=http
    HOST=gateway
    PORT=8001
    ARC_API_KEY=test-key-123
    ARC_BASE_URL=http://gateway:8001/
    OPERATION_MODE=online
    RECORDINGS_DIR=/kaggle/working/server_recording
    CAUSAL_PRIOR=/kaggle/working/ARC-AGI-3-Agents/agents/causal/prior.json
    CAUSAL_MAX_ACTIONS=100000
    ```
    (**sem** `CAUSAL_PRIOR_SAVE` → read-only; `prior.json` já gravado pelo loop
    de `FILES`.)
  - roda `!cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent causalobject`.
- **Cell 2 (code):** ramo não-rerun → grava `submission.parquet` dummy
  (`pd.DataFrame([['1_0','1',True,1]], columns=['row_id','game_id','end_of_game','score'])`).

O `agents/__init__.py` enxuto importa `.swarm` e `.agent` (base, offline-safe) e
**só** o `CausalObjectAgent`, evitando o `__init__` original que puxa
`langgraph`/`smolagents`.

### 3. `prior.json` (6a)

**Realidade descoberta:** rodar TODOS os jogos num único swarm dispara **429
Too Many Requests** na API pública (muitas requests concorrentes). Adaptação
(mesma intenção — cobrir os jogos — dentro do limite): pré-treinar em **lotes
pequenos sequenciais** (2–4 jogos por vez), acumulando via o mecanismo de
persistência do v6 (`load_shared_once` funde o disco no init; `cleanup` salva):

```bash
CAUSAL_PRIOR=agents/causal/prior.json CAUSAL_PRIOR_SAVE=1 \
  uv run main.py --agent=causalobject --game=<id1>,<id2>
# repetir com outros ids; cada run acumula sobre o anterior
```

`prior.json` commitado. O build-script o embute se presente.

## Fluxo de dados (eval Kaggle)

rerun detectado → wheels já instaladas → espera gateway → copia repo da
competição → injeta nosso pacote + `prior.json` → `.env` roteia pro gateway com
`CAUSAL_PRIOR` (read-only) e `CAUSAL_MAX_ACTIONS` alto → `main.py` roda o
`Swarm` em todos os jogos; cada agente carrega o prior pré-treinado (warm-start)
e joga até o wall-clock.

## Erros e casos de borda

- **`prior.json` ausente:** build não embute; `.env` ainda aponta `CAUSAL_PRIOR`
  para um caminho inexistente → `load_prior` retorna `None` → no-op (roda sem
  warm-start). Sem crash.
- **Base64:** robusto a qualquer conteúdo de fonte (aspas triplas, backslashes).
- **Idempotência do build:** rodar o build 2× produz o mesmo `submission.ipynb`
  (dado o mesmo fonte) — determinístico.
- **`CAUSAL_PRIOR_SAVE` nunca setado na eval** → nunca escreve o prior (Kaggle
  read-only), sem concorrência de disco.
- **Sanidade offline:** o pacote só importa numpy/arcengine/stdlib; o
  `__init__` enxuto evita as deps pesadas.

## Testes (TDD)

Build-script testável localmente (`tests/kaggle/test_build_notebook.py`):

1. **`read_sources`** embute os 9 módulos (base64 decodifica de volta ao fonte
   exato); inclui `prior.json` quando presente, omite quando ausente
   (`tmp_path` com uma cópia mínima do pacote).
2. **`build_notebook`** produz nbformat v4 válido (`json.dumps` ok; chaves
   `cells`/`nbformat`/`metadata`; ≥3 células de código).
3. **Conteúdo das células:** cell 0 tem `pip install --no-index`; cell 1 tem
   `KAGGLE_IS_COMPETITION_RERUN`, `causalobject`, `CAUSAL_PRIOR`,
   `CAUSAL_MAX_ACTIONS`, e **não** contém `CAUSAL_PRIOR_SAVE`; o `__init__`
   enxuto importa `CausalObjectAgent` e não importa `langgraph`.
4. **`main()`** grava `kaggle/submission.ipynb` e o arquivo é JSON carregável.

`MAX_ACTIONS` por env (`tests/causal/test_agent_max_actions.py`):

5. Com `CAUSAL_MAX_ACTIONS=5`, um agente recém-init tem `self.MAX_ACTIONS == 5`;
   sem a env, `== 80`.

6. **Regressão:** os 109 testes v1–v6 seguem verdes.

## Fora de escopo

- Refino coarse-to-fine, perseguição de meta, features de prior mais ricas.
- Upload do notebook no Kaggle (ação manual na conta do usuário) — entregamos o
  `.ipynb` gerado e o `build_notebook.py`.

## Critério de pronto

- `agents/causal/prior.json` commitado (pré-treino em todos os jogos).
- `MAX_ACTIONS` respeita `CAUSAL_MAX_ACTIONS`.
- `kaggle/build_notebook.py` gera um `kaggle/submission.ipynb` válido,
  auto-contido, com o pacote + prior embutidos e os envs corretos.
- 109 testes v1–v6 + novos verdes.
- Notebook abre como JSON válido e as células contêm a mecânica de submissão
  completa.
