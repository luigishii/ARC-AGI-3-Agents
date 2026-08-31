# Harness OFFLINE de de-blinding — Design

**Data:** 2026-08-31
**Status:** aprovado (brainstorming)
**Autor:** sessão Claude Code (ARC-AGI-3)

## Problema

Estamos **cegos** na avaliação Kaggle: os logs da rerun de submissão são ocultos, o
output não baixa, e o `causal_phase2.json` só existe na rerun escondida. Temos 1
submissão/dia e o score agregado (0.09) é o único sinal. Não conseguimos **ver** o
agente jogando — se o Qwen3-32B sobe, quais metas gera, se `levels_completed` sobe.

A tentativa anterior (eval público no editor, `build_eval_notebook.py`) está **morta**:
o RTX Pro 6000 (único GPU que carrega o 32B, 61GB/102GB) no Kaggle **só liga com
internet OFF**, mas a API pública precisa de internet ON. Incompatível.

## Objetivo

Um **canal de medição observável** que roda o agente completo (Qwen3-32B + flags
Fase-2) **localmente, sem internet**, no editor do Kaggle (RTX Pro 6000 offline), com
**logs visíveis**, pra ver `levels_completed` subir (ou não) nos jogos — sem gastar
submissão diária.

### Ressalva de escopo (aceita pelo usuário)

O harness offline roda **só os jogos PÚBLICOS** (os que vêm com o `environment_files`
da competição). A eval privada **nunca** está disponível offline (é o que a competição
esconde). Isso é de-blinding real: ver o 32B jogar + níveis subindo + decisões passo a
passo, nos jogos de treino. A nota privada continua proxy.

## Fato habilitador (verificado no pacote real)

O harness **já suporta OFFLINE nativamente** — não precisa de driver novo:

- `arc_agi.OperationMode.OFFLINE` existe ("local only", sem HTTP).
- `agents/swarm.py:55` — `self._arc = Arcade()` lê `OPERATION_MODE` do env
  (`normal`/`online`/`offline`).
- `agents/swarm.py:84` — `arc_env = self._arc.make(g, ...)`; em OFFLINE retorna
  `LocalEnvironmentWrapper` (in-process).
- `agents/swarm.py:110` — já ramifica em `operation_mode == ONLINE` pro scorecard →
  OFFLINE é modo de 1ª classe.
- `agents/agent.py` — `Agent` recebe `arc_env: EnvironmentWrapper` e chama
  `arc_env.step()`; a mesma interface nos dois modos. O `main()` do agente dirige o
  loop `step()` local sem tocar HTTP.
- `Arcade.__init__` — `OPERATION_MODE`, `ENVIRONMENTS_DIR`, `ARC_BASE_URL` todos vêm de
  env vars.
- OFFLINE descobre jogos varrendo `ENVIRONMENTS_DIR` por `metadata.json`.
- `FrameDataRaw` tem `levels_completed`, `win_levels`, `state` → o sinal que queremos.

Conclusão: `main.py --agent causalobject --game <g>` roda offline **só trocando o
`.env`**. A premissa anterior ("precisa de driver novo, não o loop HTTP do main.py")
estava errada — o `main.py`/Swarm abstraem o transporte e OFFLINE já está ligado.

## Arquitetura

**Aproximação A (aprovada):** troca de `.env` no harness existente. Zero driver novo.
Reusa `main.py`/`Swarm`/`Agent`/scorecard já verdes. Único código novo = o build-script
do notebook + testes.

### Componente único: `kaggle/build_offline_notebook.py`

Irmão de `build_eval_notebook.py`. Reusa de `build_notebook.py` (DRY):
`WHEELS`, `REPO`, `COMP_REPO`, `TRIMMED_INIT`, `MODEL_DATASET_PATH`, `read_sources`,
`_cell`, `_repo_root`. Gera `kaggle/offline.ipynb` (nbformat v4).

**Aditivo:** não toca `agent.py`, `submission.ipynb`, `build_notebook.py`,
`build_eval_notebook.py` (só importa do primeiro).

### `.env` offline (`OFFLINE_ENV`)

Difere do eval público: **sem** `SCHEME/HOST/PORT/ARC_BASE_URL` de HTTP, **sem**
`ARC_API_KEY` (offline não chama API → nem precisa de Kaggle Secret), **sem** gate de
rerun. Conteúdo:

```
DEBUG=True
OPERATION_MODE=offline
ENVIRONMENTS_DIR=<descoberto em runtime>
RECORDINGS_DIR=/kaggle/working/recordings
CAUSAL_PRIOR=<REPO>/agents/causal/prior.json
CAUSAL_MAX_ACTIONS=200
CAUSAL_LLM=1
CAUSAL_TYPED=1
CAUSAL_ETA=1
CAUSAL_IW=1
QWEN_MODEL_PATH=<MODEL_DATASET_PATH>
```

`ENVIRONMENTS_DIR` é o único valor descoberto em runtime (cell1) — o `.env` é escrito
concatenando o dir achado. `CAUSAL_MAX_ACTIONS=200` dá chance de completar nível sem
runs infinitos (tunável). Não hardcodamos os IDs de jogo: rodamos o que
`get_environments()` retornar.

### Células do notebook

**Cell 0 — pip offline:**
`!pip install --no-index --find-links <WHEELS> arc-agi python-dotenv`

**Cell 1 — setup + descoberta + run:**
1. `copytree(COMP_REPO → REPO, dirs_exist_ok=True)`.
2. Grava os `FILES` base64 (`agents/causal/*.py` + `prior.json`) em REPO.
3. Grava `TRIMMED_INIT` em `REPO/agents/__init__.py`.
4. **Descobre `ENVIRONMENTS_DIR`:** glob no input da competição por `metadata.json`
   (`/kaggle/input/**/metadata.json`, recursive) → dir pai comum. **Se 0 achados,
   imprime erro alto (`RuntimeError` / print de bloco visível) e NÃO segue** — sinaliza
   que os `.py` dos jogos não vêm no container.
5. Grava o `.env` offline concatenando `ENVIRONMENTS_DIR=<dir>`.
6. **Lista jogos:** constrói `Arcade(operation_mode=OFFLINE, environments_dir=<dir>)`,
   chama `get_environments()`, imprime os IDs jogáveis (de-blinding: usuário vê o que
   roda).
7. **Loop** nos jogos descobertos: `os.system('cd REPO && MPLBACKEND=agg python main.py
   --agent causalobject --game ' + gid)` — log ao vivo no editor.

**Cell 2 — diagnóstico:**
- Imprime cada `causal_phase2.json` em `/kaggle/working/**`.
- Lista os `recordings/*.jsonl`.
- Resumo por jogo de `levels_completed` (do scorecard/`causal_phase2.json` quando
  presente).

## Fluxo de dados

```
COMP_REPO (jogos + harness) ──copytree──▶ REPO (gravável)
nossos módulos base64 ──────────────────▶ REPO/agents/causal/*.py + prior.json
TRIMMED_INIT ───────────────────────────▶ REPO/agents/__init__.py
glob metadata.json ─────────────────────▶ ENVIRONMENTS_DIR ─▶ .env
Arcade(OFFLINE).get_environments() ─────▶ [game ids]
main.py --game <g> ──(Swarm→Arcade.make OFFLINE→LocalEnvironmentWrapper)──▶
  agent.main() loop step() ─────────────▶ FrameDataRaw{levels_completed} ─▶ logs
causal_phase2.json + recordings ────────▶ Cell 2 diagnóstico
```

## Tratamento de erro

- **0 jogos descobertos** (glob vazio): erro alto e visível na Cell 1; não roda o loop.
  É a única incógnita que só o Kaggle resolve (se a competição não empacota os `.py`
  dos jogos). Falhar alto > silêncio confuso.
- **Qwen degrada pro Null** (pesos ausentes/incompatíveis): o log de boot
  `[causal] LLM ativo: vllm|hf|null` (já existente no `llm.py`) revela; o run continua
  no fallback determinístico.
- **Um jogo trava/crasha:** `os.system` por jogo isola — os outros seguem.

## Testes (`tests/kaggle/test_build_offline_notebook.py`)

Espelham o estilo de `test_build_eval_notebook.py`. Carregam o módulo via
`importlib.util.spec_from_file_location`. `ROOT` = 3 dirs acima do arquivo de teste.

1. `test_valid_nbformat` — `build_offline_notebook(sources)` gera nbformat v4 válido
   (3 células code).
2. `test_modules_embedded` — todos os `MODULES` + `prior.json` aparecem base64 no
   notebook.
3. `test_env_offline_not_http` — `.env` contém `OPERATION_MODE=offline`; **não** contém
   `arcprize.org` nem `HOST=gateway` nem `SCHEME=`.
4. `test_no_rerun_gate` — string do notebook não contém `KAGGLE_IS_COMPETITION_RERUN`.
5. `test_no_api_key` — não contém `ARC_API_KEY` hardcoded nem `UserSecretsClient`
   (offline não precisa).
6. `test_llm_flags_present` — `.env` contém `CAUSAL_LLM=1`, `CAUSAL_TYPED=1`,
   `CAUSAL_ETA=1`, `CAUSAL_IW=1`, `QWEN_MODEL_PATH`.
7. `test_env_discovery_present` — a Cell 1 contém glob por `metadata.json` e escreve
   `ENVIRONMENTS_DIR` no `.env`.
8. `test_get_environments_listing` — a Cell 1 contém `get_environments()` (listagem de
   jogos jogáveis).

**Restrição global:** manter a suíte inteira verde (295 → ~303). `numpy`/stdlib apenas.
Rodar ao vivo é no Kaggle (fora do plano).

## Runbook do usuário (pós-implementação, no Kaggle)

1. Anexar a competição ARC Prize 2026 + o dataset do Qwen3-32B ao notebook do editor.
2. `uv run python kaggle/build_offline_notebook.py` → gera `kaggle/offline.ipynb`.
3. Colar o `offline.ipynb` num notebook do editor Kaggle; GPU = RTX Pro 6000;
   **internet OFF**.
4. Rodar todas as células. Observar: log de boot do LLM, lista de jogos, `main.py`
   jogando com log ao vivo, `levels_completed` por jogo no diagnóstico.
5. Se Cell 1 disser "0 environments": os `.py` dos jogos não vêm no container →
   reportar (decide se há outra fonte de jogos públicos).

## Fora de escopo

- Rodar a eval privada offline (impossível — jogos ocultos).
- Mudanças no world-model (próxima alavanca, após o harness abrir os olhos).
- Alterar `main.py`/Swarm/agent (o offline já é suportado).
