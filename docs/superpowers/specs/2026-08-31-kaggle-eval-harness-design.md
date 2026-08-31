# Harness de De-blinding (editor Kaggle) — Design

**Data:** 2026-08-31
**Status:** aprovado (brainstorming)
**Autor:** sessão CausalObjectAgent

## Problema

Depois do primeiro score não-zero (**0.09** com Qwen3-32B), a próxima alavanca é
maturar o world-model. Mas iterar está travado por duas restrições que se combinam mal:

1. **Cegueira na eval.** Os logs da rerun de submissão são ocultos e o output não
   baixa. Só enxergamos o **score agregado**.
2. **1 submissão/dia** + **sem GPU local** (o Qwen3-32B só roda no Kaggle).

Consequência: uma mudança de world-model validada "no escuro" que derrube o score
não diz **qual parte** quebrou. Iterar assim = 1 tentativa cega por dia.

## Solução

No **editor** do Kaggle (diferente da submissão) a **internet está ON e os logs são
visíveis**. Vamos rodar o **agente completo (32B + todas as flags) contra a API
pública** dentro do editor, com **logs ao vivo + diagnóstico impresso**. Isso
transforma "1 tiro cego/dia" em "iteração observável no editor → só o melhor vira
submissão". Os jogos públicos ≠ os privados da eval, mas **a mecânica é a mesma**.

## Escopo

**Dentro:** um build-script que gera um notebook de avaliação auto-contido rodável no
editor, apontando pra API pública, com observabilidade (logs + diagnóstico).

**Fora (YAGNI):** GIF/montagem visual (já existe `analysis/replay.py` offline se
precisar depois); rodar todos os jogos de uma vez; qualquer fine-tuning; mudanças no
`submission.ipynb` ou no agente. Este projeto entrega **só o canal observável**.

## Arquitetura

**Artefato:** `kaggle/build_eval_notebook.py` (build-script) → gera `kaggle/eval.ipynb`.

**DRY — reúso do `kaggle/build_notebook.py`:** importa e reusa as peças compartilhadas
`read_sources`, `MODULES`, `TRIMMED_INIT`, `MODEL_DATASET_PATH`, `_cell`. O eval-builder
difere do submission-builder em **exatamente dois pontos**: o bloco `.env` e a cell de
run. Nada é duplicado; se `MODULES`/`MODEL_DATASET_PATH` mudam no submission, o eval
herda automaticamente.

**Três diferenças pro `submission.ipynb`:**

1. **Sem gate de rerun.** A cell de run executa **incondicionalmente** (o editor tem
   internet). O submission gateia em `KAGGLE_IS_COMPETITION_RERUN`; o eval não.
2. **`.env` aponta pra API PÚBLICA**, não pro gateway.
3. **Roda um punhado de jogos em sequência** (default `["vc33", "ls20"]`), pra evitar o
   **429 (rate limit)** que já pegamos rodando tudo de uma vez.

## Componentes

### `kaggle/build_eval_notebook.py`

Módulo com:

- **`EVAL_ENV`** (str) — bloco `.env` para o editor. Conteúdo **verbatim**:
  ```
  DEBUG=True
  SCHEME=https
  HOST=arcprize.org
  PORT=443
  ARC_BASE_URL=https://arcprize.org/
  OPERATION_MODE=online
  RECORDINGS_DIR=/kaggle/working/recordings
  CAUSAL_PRIOR=<REPO>/agents/causal/prior.json
  CAUSAL_MAX_ACTIONS=80
  CAUSAL_LLM=1
  CAUSAL_TYPED=1
  CAUSAL_ETA=1
  CAUSAL_IW=1
  QWEN_MODEL_PATH=<MODEL_DATASET_PATH>
  ```
  `<REPO>` = a constante `REPO` importada do `build_notebook.py`
  (`/kaggle/working/ARC-AGI-3-Agents`); `<MODEL_DATASET_PATH>` idem.
  **A `ARC_API_KEY` NÃO aparece aqui** — é injetada em runtime (ver cell de run).

- **`GAMES`** (list[str]) — jogos default a rodar em sequência: `["vc33", "ls20"]`.

- **`build_eval_notebook(sources)`** — retorna o dict nbformat v4 com 3 cells:
  - **cell0** — `!pip install --no-index --find-links <WHEELS> arc-agi python-dotenv`
    (mesma linha do submission; `WHEELS` importado do `build_notebook.py`).
  - **cell1** — a cell de run (ver abaixo).
  - **cell2** — a cell de diagnóstico (ver abaixo).

- **`main()`** — `sources = read_sources(<repo_root>)`; `nb = build_eval_notebook(sources)`;
  grava em `kaggle/eval.ipynb` com `json.dump(nb, f, indent=1)`; imprime o caminho e a
  contagem de arquivos embutidos.

### Cell de run (editor, SEM gate de rerun)

Pseudo-código (o real é string embutida, análogo à cell1 do submission mas sem o
`if os.getenv('KAGGLE_IS_COMPETITION_RERUN')`):

```python
import os, shutil, base64
from kaggle_secrets import UserSecretsClient
key = UserSecretsClient().get_secret("ARC_API_KEY")
shutil.copytree(COMP_REPO, REPO, dirs_exist_ok=True)          # repo da competição (mounted no editor)
FILES = {...}                                                  # nosso pacote base64 (read_sources)
for rel, b64 in FILES.items():
    dst = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, 'wb').write(base64.b64decode(b64))
open(REPO + '/agents/__init__.py', 'w').write(TRIMMED_INIT)   # __init__ enxuto (offline-safe)
open(REPO + '/.env', 'w').write(EVAL_ENV + 'ARC_API_KEY=' + key + '\n')  # key só em runtime
for g in GAMES:
    os.system(f"cd {REPO} && MPLBACKEND=agg python main.py --agent causalobject --game {g}")
```

`COMP_REPO`, `REPO`, `TRIMMED_INIT` importados do `build_notebook.py`.

### Cell de diagnóstico

Depois dos jogos, imprime os sinais que nos de-cegam:

```python
import json, glob
for p in sorted(glob.glob('/kaggle/working/**/causal_phase2.json', recursive=True)):
    print("=== causal_phase2.json ===")
    print(open(p).read())
for p in sorted(glob.glob('/kaggle/working/recordings/*.jsonl', recursive=True)):
    print(p)   # pista de níveis/ações por jogo
```

O `causal_phase2.json` traz **`llm_kind`** (confirma se o 32B subiu ou degradou pro
Null), `llm_calls`, `n_types`, `n_rules`, `reward_learned`, `eta_rows`.

## Fluxo de dados

editor com internet ON → cell0 instala wheels → cell1 lê o secret, monta o repo +
nosso pacote + `.env` público, roda `main.py --game G` por jogo (o **stdout** mostra
ao vivo: cada ação com id+x,y+**reasoning**, level-ups, metas do LLM) → o agente grava
`causal_phase2.json` + recordings em `/kaggle/working` → cell2 imprime o diagnóstico.

## Tratamento de erros

- **Secret ausente** — se `get_secret("ARC_API_KEY")` levantar, a cell falha com
  mensagem clara; o runbook manda cadastrar o secret. (Não mascarar — queremos saber.)
- **429 (rate limit)** — mitigado rodando **sequencial** (não em paralelo). Se ainda
  ocorrer, o usuário reduz `GAMES` a 1 jogo. Não vamos adicionar retry/backoff agora
  (YAGNI; a mecânica de sequência já resolve o caso comum).
- **LLM degrada pro Null** — não é erro fatal; o `causal_phase2.json` expõe
  `llm_kind="null"`, que é justamente o sinal que queremos ver.

## Testes (offline, neste ambiente)

Espelham `tests/kaggle/test_build_notebook.py`, em
`tests/kaggle/test_build_eval_notebook.py`:

1. **nbformat válido** — `build_eval_notebook(read_sources(root))` produz dict com
   `nbformat == 4` e 3 cells de código.
2. **Módulos embutidos** — as fontes base64 do nosso pacote aparecem na cell de run
   (reusa `read_sources`; ao menos `agent.py` e `llm.py` presentes).
3. **`.env` público, não gateway** — a cell de run contém `HOST=arcprize.org` e
   `arcprize.org/` e **NÃO** contém `HOST=gateway`.
4. **Sem gate de rerun** — a cell de run **NÃO** contém `KAGGLE_IS_COMPETITION_RERUN`.
5. **Key não hardcoded** — o texto do notebook **NÃO** contém `ARC_API_KEY=` seguido de
   valor literal; contém a leitura via `UserSecretsClient`/`get_secret("ARC_API_KEY")`.
6. **Flags do LLM presentes** — a cell de run contém `CAUSAL_LLM=1` e
   `QWEN_MODEL_PATH=`.

O **run ao vivo é no Kaggle** (não testável aqui, igual ao submission): validação real =
o usuário roda `eval.ipynb` no editor e lê os logs.

Manter os **289 testes** existentes verdes (o eval-builder é aditivo; não toca o
submission-builder além de importar dele).

## Runbook do usuário (setup 1×)

1. **Add-ons → Secrets** do notebook → adicionar secret `ARC_API_KEY` com a chave da
   ARC, e habilitar pro notebook.
2. Anexar (Input): dataset **Qwen3-32B** + a **competição** ARC-AGI-3.
3. Regenerar: `uv run python kaggle/build_eval_notebook.py`.
4. Abrir `kaggle/eval.ipynb` no editor, **Run All**, assistir os logs.

## Restrições globais

- **numpy/stdlib** no núcleo; o build-script é stdlib puro (`base64`, `json`, `os`).
- **Aditivo** — nenhuma mudança no `agent.py`, no `submission.ipynb`, ou no
  `build_notebook.py` (só **import** dele).
- Manter 289 testes verdes; adicionar os novos em `tests/kaggle/`.
- A `ARC_API_KEY` **nunca** entra no arquivo `eval.ipynb` versionado.
