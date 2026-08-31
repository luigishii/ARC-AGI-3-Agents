# Harness OFFLINE de de-blinding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gerar `kaggle/offline.ipynb` que roda o agente completo (Qwen3-32B + flags Fase-2) contra o Arcade OFFLINE localmente no editor Kaggle, com logs visíveis, pra observar `levels_completed` por jogo público sem gastar submissão.

**Architecture:** Aproximação A — só troca de `.env`, sem driver novo. O `Swarm` já lê `OPERATION_MODE` do env e `Arcade.make()` OFFLINE devolve `LocalEnvironmentWrapper` in-process; `main.py --agent causalobject --game <g>` roda offline apenas com `OPERATION_MODE=offline` + `ENVIRONMENTS_DIR` no `.env`. Um único build-script novo (`kaggle/build_offline_notebook.py`) reusa as peças do `build_notebook.py` (DRY) e emite o notebook de 3 células.

**Tech Stack:** Python 3.12, stdlib (`json`, `os`, `sys`, `importlib`, `base64`, `glob`), pytest. Sem deps novas.

## Global Constraints

- Aditivo: NÃO alterar `agents/`, `main.py`, `kaggle/submission.ipynb`, `kaggle/build_notebook.py`, `kaggle/build_eval_notebook.py`. O novo módulo só **importa** de `build_notebook.py`.
- Reusar de `build_notebook.py`: `WHEELS`, `REPO`, `COMP_REPO`, `TRIMMED_INIT`, `MODEL_DATASET_PATH`, `read_sources`, `_cell`, `_repo_root`.
- `.env` offline: contém `OPERATION_MODE=offline`; NÃO contém `arcprize.org`, `HOST=gateway`, nem `SCHEME=`.
- Sem gate de rerun: a string do notebook NÃO contém `KAGGLE_IS_COMPETITION_RERUN`.
- Sem chave: NÃO contém `ARC_API_KEY` hardcoded nem `UserSecretsClient` (offline não chama API).
- Flags LLM no `.env`: `CAUSAL_LLM=1`, `CAUSAL_TYPED=1`, `CAUSAL_ETA=1`, `CAUSAL_IW=1`, `QWEN_MODEL_PATH`, `CAUSAL_MAX_ACTIONS=200`.
- Descoberta de jogos: Cell 1 faz glob por `metadata.json` e escreve `ENVIRONMENTS_DIR` no `.env`; se 0 achados, falha alto e não roda o loop.
- Listagem: Cell 1 chama `get_environments()` pra imprimir os jogos jogáveis.
- Manter a suíte inteira verde (295 → ~303). `numpy`/stdlib apenas.
- Rodar ao vivo é no Kaggle (fora do plano).

---

### Task 1: `kaggle/build_offline_notebook.py` + testes

**Files:**
- Create: `kaggle/build_offline_notebook.py`
- Test: `tests/kaggle/test_build_offline_notebook.py`

**Interfaces:**
- Consumes (de `kaggle/build_notebook.py`): `WHEELS: str`, `REPO: str`, `COMP_REPO: str`, `TRIMMED_INIT: str`, `MODEL_DATASET_PATH: str`, `read_sources(root: str) -> dict[str,str]`, `_cell(src: str) -> dict`, `_repo_root() -> str`.
- Produces: `OFFLINE_ENV: str` (o corpo do `.env` até `ENVIRONMENTS_DIR=`, que a Cell 1 completa em runtime), `build_offline_notebook(sources: dict) -> dict` (nbformat v4, 3 células code), `main() -> None` (grava `kaggle/offline.ipynb`).

O build-script é aditivo e determinístico: `build_offline_notebook(sources)` monta 3 células de string; os testes inspecionam o texto da Cell 1 (`nb["cells"][1]["source"]` joinado) e a estrutura do nbformat. Como as 8 asserções cobrem partes distintas do mesmo artefato pequeno, escrevemos todos os testes primeiro (vermelho), depois o módulo inteiro (verde) — o ciclo TDD é um só para o arquivo.

- [ ] **Step 1: Escrever o arquivo de testes (falhando)**

Criar `tests/kaggle/test_build_offline_notebook.py` com exatamente este conteúdo:

```python
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load():
    path = os.path.join(ROOT, "kaggle", "build_offline_notebook.py")
    spec = importlib.util.spec_from_file_location("build_offline_notebook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_cell():
    bn = _load()
    nb = bn.build_offline_notebook(bn.read_sources(ROOT))
    return nb, "".join(nb["cells"][1]["source"])


def _full_text():
    bn = _load()
    nb = bn.build_offline_notebook(bn.read_sources(ROOT))
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_valid_nbformat():
    bn = _load()
    nb = bn.build_offline_notebook(bn.read_sources(ROOT))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) == 3
    assert all(c["cell_type"] == "code" for c in nb["cells"])


def test_modules_embedded():
    _, run = _run_cell()
    assert "agents/causal/agent.py" in run
    assert "agents/causal/llm.py" in run
    assert "agents/causal/prior.json" in run


def test_env_offline_not_http():
    _, run = _run_cell()
    assert "OPERATION_MODE=offline" in run
    assert "arcprize.org" not in run
    assert "HOST=gateway" not in run
    assert "SCHEME=" not in run


def test_no_rerun_gate():
    assert "KAGGLE_IS_COMPETITION_RERUN" not in _full_text()


def test_no_api_key():
    txt = _full_text()
    assert "ARC_API_KEY" not in txt
    assert "UserSecretsClient" not in txt


def test_llm_flags_present():
    _, run = _run_cell()
    for flag in ("CAUSAL_LLM=1", "CAUSAL_TYPED=1", "CAUSAL_ETA=1", "CAUSAL_IW=1"):
        assert flag in run
    assert "QWEN_MODEL_PATH=" in run


def test_env_discovery_present():
    _, run = _run_cell()
    assert "metadata.json" in run
    assert "ENVIRONMENTS_DIR=" in run


def test_get_environments_listing():
    _, run = _run_cell()
    assert "get_environments()" in run
```

- [ ] **Step 2: Rodar os testes pra ver falhar**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/offline-harness && uv run pytest tests/kaggle/test_build_offline_notebook.py -q`
Expected: FAIL na coleta / `ModuleNotFoundError` ou `FileNotFoundError` (o módulo `kaggle/build_offline_notebook.py` ainda não existe).

- [ ] **Step 3: Escrever o módulo**

Criar `kaggle/build_offline_notebook.py` com exatamente este conteúdo:

```python
# kaggle/build_offline_notebook.py — gera kaggle/offline.ipynb (harness OFFLINE de
# de-blinding). Roda o agente completo (Qwen3-32B + flags) contra o Arcade OFFLINE
# no EDITOR do Kaggle (RTX Pro 6000, internet OFF), com logs visiveis. Reusa as pecas
# do build_notebook.py (DRY): so muda o .env (offline, sem chave) e as cells de run.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # torna build_notebook importavel
from build_notebook import (  # noqa: E402
    WHEELS, REPO, COMP_REPO, TRIMMED_INIT, MODEL_DATASET_PATH,
    read_sources, _cell, _repo_root,
)

# .env offline: sem HTTP, sem ARC_API_KEY (offline nao chama API). ENVIRONMENTS_DIR
# e completado em runtime pela Cell 1 (descoberto por glob de metadata.json).
OFFLINE_ENV = (
    "DEBUG=True\n"
    "OPERATION_MODE=offline\n"
    "RECORDINGS_DIR=/kaggle/working/recordings\n"
    "CAUSAL_PRIOR=" + REPO + "/agents/causal/prior.json\n"
    "CAUSAL_MAX_ACTIONS=200\n"
    "CAUSAL_LLM=1\n"
    "CAUSAL_TYPED=1\n"
    "CAUSAL_ETA=1\n"
    "CAUSAL_IW=1\n"
    "QWEN_MODEL_PATH=" + MODEL_DATASET_PATH + "\n"
)


def build_offline_notebook(sources):
    cell0 = "!pip install --no-index --find-links %s arc-agi python-dotenv\n" % WHEELS
    cell1 = (
        "import os, shutil, base64, glob\n"
        f"shutil.copytree({COMP_REPO!r}, {REPO!r}, dirs_exist_ok=True)\n"
        f"FILES = {json.dumps(sources)}\n"
        "for rel, b64 in FILES.items():\n"
        f"    dst = os.path.join({REPO!r}, rel)\n"
        "    os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
        "    with open(dst, 'wb') as f:\n"
        "        f.write(base64.b64decode(b64))\n"
        f"with open({REPO!r} + '/agents/__init__.py', 'w') as f:\n"
        f"    f.write({TRIMMED_INIT!r})\n"
        "metas = glob.glob('/kaggle/input/**/metadata.json', recursive=True)\n"
        "if not metas:\n"
        "    raise RuntimeError('0 environments: nenhum metadata.json em /kaggle/input '\n"
        "                       '-> os .py dos jogos nao vem no container offline')\n"
        "env_dir = os.path.dirname(os.path.commonprefix(metas))\n"
        "print('ENVIRONMENTS_DIR =', env_dir, '(%d metadata.json)' % len(metas))\n"
        f"with open({REPO!r} + '/.env', 'w') as f:\n"
        f"    f.write({OFFLINE_ENV!r} + 'ENVIRONMENTS_DIR=' + env_dir + '\\n')\n"
        "os.environ['OPERATION_MODE'] = 'offline'\n"
        "os.environ['ENVIRONMENTS_DIR'] = env_dir\n"
        "from arc_agi import Arcade, OperationMode\n"
        "arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)\n"
        "envs = arc.get_environments()\n"
        "games = [e.game_id for e in envs]\n"
        "print('jogos jogaveis:', games)\n"
        f"for g in games:\n"
        f"    print('=== jogando', g, '===')\n"
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
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline.ipynb")


def main():
    sources = read_sources(_repo_root())
    nb = build_offline_notebook(sources)
    out = _out_path()
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, "with", len(sources), "embedded files")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar os testes do arquivo pra ver passar**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/offline-harness && uv run pytest tests/kaggle/test_build_offline_notebook.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Rodar a suíte inteira pra garantir zero regressão**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/offline-harness && uv run pytest tests/ -q`
Expected: PASS (~303 passed; os testes pré-existentes do harness em `tests/unit/` que já falhavam na base — `agents.structs` inexistente — continuam iguais, não são nossos e não devem ter mudado de contagem).

- [ ] **Step 6: Gerar o notebook de verdade (fumaça do build-script)**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/offline-harness && uv run python kaggle/build_offline_notebook.py`
Expected: imprime `wrote .../kaggle/offline.ipynb with <N> embedded files` (N ≥ 20). Gera `kaggle/offline.ipynb`.

- [ ] **Step 7: Validar o nbformat do arquivo gerado**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/offline-harness && uv run python -c "import json; nb=json.load(open('kaggle/offline.ipynb')); assert nb['nbformat']==4 and len(nb['cells'])==3 and all(c['cell_type']=='code' for c in nb['cells']); print('OK', len(nb['cells']), 'cells')"`
Expected: `OK 3 cells`.

- [ ] **Step 8: Commit**

```bash
cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/offline-harness
git add kaggle/build_offline_notebook.py tests/kaggle/test_build_offline_notebook.py kaggle/offline.ipynb
git commit -m "feat(offline): harness OFFLINE de de-blinding (Arcade OFFLINE + Qwen3-32B, sem driver novo)"
```

---

## Self-Review

**1. Spec coverage:**
- Componente único `kaggle/build_offline_notebook.py` → Task 1 (Step 3). ✓
- `.env` offline (OPERATION_MODE=offline, sem HTTP/chave, flags LLM) → `OFFLINE_ENV` + testes `test_env_offline_not_http`/`test_no_api_key`/`test_llm_flags_present`. ✓
- Descoberta de `ENVIRONMENTS_DIR` por glob de `metadata.json` + falha alta se 0 → Cell 1 `raise RuntimeError` + `test_env_discovery_present`. ✓
- Listagem `get_environments()` → Cell 1 + `test_get_environments_listing`. ✓
- Loop nos jogos descobertos com `main.py --game <g>` → Cell 1. ✓
- Diagnóstico `causal_phase2.json` + recordings → Cell 2. ✓
- Sem gate de rerun → `test_no_rerun_gate`. ✓
- Módulos base64 embutidos → `test_modules_embedded`. ✓
- Aditivo / suíte verde → Step 5. ✓

**2. Placeholder scan:** Sem TBD/TODO. Todo código está literal.

**3. Type consistency:** `build_offline_notebook(sources)`, `read_sources(ROOT)`, `_cell`, `_repo_root`, `OFFLINE_ENV`, `main` — nomes consistentes entre módulo e testes. A Cell 1 usa `e.game_id` (campo confirmado em `EnvironmentInfo`/`FrameDataRaw.game_id`); se `get_environments()` retornar objetos sem `.game_id`, é ajuste de runtime no Kaggle (fora do plano), mas o campo `game_id` é o documentado.
