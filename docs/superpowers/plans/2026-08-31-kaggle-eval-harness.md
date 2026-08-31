# Harness de De-blinding (editor Kaggle) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Um build-script que gera `kaggle/eval.ipynb` — roda o agente completo (Qwen3-32B + flags) contra a API pública **no editor** do Kaggle, com logs visíveis + diagnóstico impresso, pra medir mudanças de world-model sem gastar a submissão diária.

**Architecture:** `kaggle/build_eval_notebook.py` importa as peças compartilhadas do `kaggle/build_notebook.py` (`WHEELS`, `REPO`, `COMP_REPO`, `TRIMMED_INIT`, `MODEL_DATASET_PATH`, `read_sources`, `_cell`, `_repo_root`) e difere só no bloco `.env` (público, sem gate de rerun, key via Kaggle Secrets) e na cell de run (roda `main.py --game G` sequencial). Puramente aditivo.

**Tech Stack:** Python stdlib (`base64`, `json`, `os`, `sys`), nbformat v4, pytest.

## Global Constraints

- **numpy/stdlib** apenas; o build-script é stdlib puro.
- **Aditivo** — NÃO modificar `agents/causal/agent.py`, `kaggle/submission.ipynb`, nem `kaggle/build_notebook.py` (só **importar** deste último).
- A `ARC_API_KEY` **nunca** entra no arquivo `eval.ipynb` versionado — é lida via `UserSecretsClient().get_secret("ARC_API_KEY")` em runtime.
- `.env` do eval aponta pra API **pública** (`HOST=arcprize.org`), **não** pro gateway.
- Cell de run **sem** gate `KAGGLE_IS_COMPETITION_RERUN`.
- `GAMES = ["vc33", "ls20"]`; `CAUSAL_MAX_ACTIONS=80`; flags `CAUSAL_LLM=1 CAUSAL_TYPED=1 CAUSAL_ETA=1 CAUSAL_IW=1`.
- Manter os **289 testes** existentes verdes.

---

### Task 1: `build_eval_notebook.py` + testes

**Files:**
- Create: `kaggle/build_eval_notebook.py`
- Create: `tests/kaggle/test_build_eval_notebook.py`
- Reference (não modificar): `kaggle/build_notebook.py`

**Interfaces:**
- Consumes (import de `kaggle/build_notebook.py`): `WHEELS: str`, `REPO: str`, `COMP_REPO: str`, `TRIMMED_INIT: str`, `MODEL_DATASET_PATH: str`, `read_sources(root) -> dict[str,str]`, `_cell(src: str) -> dict`, `_repo_root() -> str`.
- Produces: `EVAL_ENV: str`, `GAMES: list[str]`, `build_eval_notebook(sources: dict) -> dict`, `main() -> None`.

- [ ] **Step 1: Escrever os testes que falham**

Criar `tests/kaggle/test_build_eval_notebook.py`:

```python
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load():
    path = os.path.join(ROOT, "kaggle", "build_eval_notebook.py")
    spec = importlib.util.spec_from_file_location("build_eval_notebook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_cell():
    bn = _load()
    nb = bn.build_eval_notebook(bn.read_sources(ROOT))
    return nb, "".join(nb["cells"][1]["source"])


def test_valid_nbformat():
    bn = _load()
    nb = bn.build_eval_notebook(bn.read_sources(ROOT))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 3
    assert all(c["cell_type"] == "code" for c in nb["cells"])


def test_modules_embedded():
    _, run = _run_cell()
    assert "agents/causal/agent.py" in run
    assert "agents/causal/llm.py" in run


def test_env_public_not_gateway():
    _, run = _run_cell()
    assert "arcprize.org" in run
    assert "HOST=gateway" not in run


def test_no_rerun_gate():
    _, run = _run_cell()
    assert "KAGGLE_IS_COMPETITION_RERUN" not in run


def test_key_via_secrets_not_hardcoded():
    _, run = _run_cell()
    assert "UserSecretsClient" in run
    assert "get_secret('ARC_API_KEY')" in run or 'get_secret("ARC_API_KEY")' in run


def test_llm_flags_present():
    _, run = _run_cell()
    assert "CAUSAL_LLM=1" in run
    assert "QWEN_MODEL_PATH=" in run
```

- [ ] **Step 2: Rodar os testes pra ver falhar**

Run: `uv run pytest tests/kaggle/test_build_eval_notebook.py -q`
Expected: FAIL (ModuleNotFoundError / arquivo `kaggle/build_eval_notebook.py` não existe).

- [ ] **Step 3: Implementar `kaggle/build_eval_notebook.py`**

```python
# kaggle/build_eval_notebook.py — gera kaggle/eval.ipynb (harness de de-blinding).
# Roda o agente completo (Qwen3-32B + flags) contra a API PUBLICA no EDITOR do
# Kaggle, com logs visiveis. Reusa as pecas do build_notebook.py (DRY): so muda o
# .env (publico, key via Kaggle Secrets) e a cell de run (sem gate de rerun).
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # torna build_notebook importavel
from build_notebook import (  # noqa: E402
    WHEELS, REPO, COMP_REPO, TRIMMED_INIT, MODEL_DATASET_PATH,
    read_sources, _cell, _repo_root,
)

GAMES = ["vc33", "ls20"]  # sequencial evita 429 (rate limit)

# .env do editor: API publica (nao o gateway). ARC_API_KEY NAO entra aqui — a cell
# de run le do Kaggle Secrets e concatena em runtime.
EVAL_ENV = (
    "DEBUG=True\n"
    "SCHEME=https\n"
    "HOST=arcprize.org\n"
    "PORT=443\n"
    "ARC_BASE_URL=https://arcprize.org/\n"
    "OPERATION_MODE=online\n"
    "RECORDINGS_DIR=/kaggle/working/recordings\n"
    "CAUSAL_PRIOR=" + REPO + "/agents/causal/prior.json\n"
    "CAUSAL_MAX_ACTIONS=80\n"
    "CAUSAL_LLM=1\n"
    "CAUSAL_TYPED=1\n"
    "CAUSAL_ETA=1\n"
    "CAUSAL_IW=1\n"
    "QWEN_MODEL_PATH=" + MODEL_DATASET_PATH + "\n"
)


def build_eval_notebook(sources):
    cell0 = "!pip install --no-index --find-links %s arc-agi python-dotenv\n" % WHEELS
    cell1 = (
        "import os, shutil, base64\n"
        "from kaggle_secrets import UserSecretsClient\n"
        "key = UserSecretsClient().get_secret('ARC_API_KEY')\n"
        f"shutil.copytree({COMP_REPO!r}, {REPO!r}, dirs_exist_ok=True)\n"
        f"FILES = {json.dumps(sources)}\n"
        "for rel, b64 in FILES.items():\n"
        f"    dst = os.path.join({REPO!r}, rel)\n"
        "    os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
        "    with open(dst, 'wb') as f:\n"
        "        f.write(base64.b64decode(b64))\n"
        f"with open({REPO!r} + '/agents/__init__.py', 'w') as f:\n"
        f"    f.write({TRIMMED_INIT!r})\n"
        f"with open({REPO!r} + '/.env', 'w') as f:\n"
        f"    f.write({EVAL_ENV!r} + 'ARC_API_KEY=' + key + '\\n')\n"
        f"for g in {GAMES!r}:\n"
        f"    os.system('cd {REPO} && MPLBACKEND=agg python main.py --agent causalobject --game ' + g)\n"
    )
    cell2 = (
        "import glob\n"
        "for p in sorted(glob.glob('/kaggle/working/**/causal_phase2.json', recursive=True)):\n"
        "    print('=== causal_phase2.json ===')\n"
        "    print(open(p).read())\n"
        "for p in sorted(glob.glob('/kaggle/working/recordings/*.jsonl')):\n"
        "    print(p)\n"
    )
    return {
        "cells": [_cell(cell0), _cell(cell1), _cell(cell2)],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def _out_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.ipynb")


def main():
    sources = read_sources(_repo_root())
    nb = build_eval_notebook(sources)
    out = _out_path()
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, "with", len(sources), "embedded files")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar os testes novos + suite completa**

Run: `uv run pytest tests/kaggle/test_build_eval_notebook.py -q`
Expected: PASS (6 passed).

Run: `uv run pytest tests/causal tests/kaggle -q`
Expected: PASS (295 passed = 289 + 6).

- [ ] **Step 5: Gerar o notebook e conferir**

Run: `uv run python kaggle/build_eval_notebook.py`
Expected: imprime `wrote .../kaggle/eval.ipynb with 21 embedded files`.

Run: `uv run python -c "import json; nb=json.load(open('kaggle/eval.ipynb')); r=''.join(nb['cells'][1]['source']); print('publico:', 'arcprize.org' in r); print('sem gateway:', 'HOST=gateway' not in r); print('sem gate rerun:', 'KAGGLE_IS_COMPETITION_RERUN' not in r); print('secret:', 'get_secret' in r)"`
Expected: todas `True`.

- [ ] **Step 6: Commit**

```bash
git add kaggle/build_eval_notebook.py kaggle/eval.ipynb tests/kaggle/test_build_eval_notebook.py
git commit -m "feat(eval): harness de de-blinding no editor Kaggle (32B vs API publica)"
```

---

## Self-Review

**1. Spec coverage:**
- Artefato `build_eval_notebook.py` reusando `build_notebook.py` → Task 1 (import block). ✅
- Sem gate de rerun → `test_no_rerun_gate` + cell1 sem `if KAGGLE_IS...`. ✅
- `.env` público → `EVAL_ENV` + `test_env_public_not_gateway`. ✅
- Key via Kaggle Secrets em runtime → cell1 `UserSecretsClient` + `test_key_via_secrets_not_hardcoded`. ✅
- `GAMES=["vc33","ls20"]` sequencial → constante + loop `os.system`. ✅
- Cell de diagnóstico (causal_phase2.json + recordings) → cell2. ✅
- 6 testes espelhando `test_build_notebook.py` → Task 1 Step 1. ✅
- Aditivo (não toca agent/submission/build_notebook) → só import; Global Constraints. ✅
- Run ao vivo no Kaggle fora do plano → coberto pelo runbook da spec, não é task. ✅

**2. Placeholder scan:** nenhum "TBD/TODO"; todo código presente. ✅

**3. Type consistency:** `build_eval_notebook(sources) -> dict`, `read_sources(root) -> dict`, `_cell(src) -> dict`, `EVAL_ENV: str`, `GAMES: list[str]` — consistentes entre plano e testes. O `EVAL_ENV!r` (repr) embute `arcprize.org`/`CAUSAL_LLM=1`/`QWEN_MODEL_PATH=` no texto da cell1 → os asserts dos testes batem. ✅
