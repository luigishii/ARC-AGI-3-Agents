# Fase 6 — Notebook de submissão Kaggle · Plano

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gerar um notebook Kaggle auto-contido que injeta o pacote `agents/causal/` + `prior.json` no ambiente da eval e roda o agente `causalobject`, e tornar `MAX_ACTIONS` configurável por env.

**Architecture:** Mudança pontual em `agents/causal/agent.py` (MAX_ACTIONS por env) + novo `kaggle/build_notebook.py` (build-script que base64-encoda o pacote e emite `kaggle/submission.ipynb` nbformat v4). Demais módulos do agente não mudam. `prior.json` vem do pré-treino ao vivo (fora do plano de código).

**Tech Stack:** Python 3.12, numpy/stdlib puro (`base64`, `json`, `os`, `shutil`), pytest. Sem LLM/GPU. Kaggle-submittable.

## Global Constraints

- Numpy/stdlib puro; nenhuma dependência nova; nada de LLM/GPU.
- `MAX_ACTIONS` por instância: `int(os.environ.get("CAUSAL_MAX_ACTIONS", type(self).MAX_ACTIONS))`; classe mantém `MAX_ACTIONS = 80`.
- Notebook: cell0 `pip install --no-index` (wheels `arc-agi`, `python-dotenv`); cell1 ramo rerun; cell2 dummy `submission.parquet`.
- `.env` do notebook: gateway + `CAUSAL_PRIOR=<prior shipado>` + `CAUSAL_MAX_ACTIONS=100000`, **sem** `CAUSAL_PRIOR_SAVE`.
- `agents/__init__.py` enxuto importa só `causalobject`, com `load_dotenv()` ANTES do import do agente.
- Código dos módulos embutido em **base64** (à prova de colisão).
- Não alterar `policy/perception/hud/causal_model/novelty/transfer/instrumentation`.
- Os 109 testes v1–v6 devem seguir verdes.

---

### Task 1: `MAX_ACTIONS` por env (`agent.py`)

**Files:**
- Modify: `agents/causal/agent.py` (`_init_causal_state`)
- Test: `tests/causal/test_agent_max_actions.py` (novo)

**Interfaces:**
- Consumes: `CausalObjectAgent` (v6) com `MAX_ACTIONS = 80` de classe.
- Produces: `self.MAX_ACTIONS` por instância lido de `CAUSAL_MAX_ACTIONS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_agent_max_actions.py
from agents.causal.agent import CausalObjectAgent


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a._cleanup = False
    a._init_causal_state()
    return a


def test_default_max_actions_is_80(monkeypatch):
    monkeypatch.delenv("CAUSAL_MAX_ACTIONS", raising=False)
    a = _agent()
    assert a.MAX_ACTIONS == 80


def test_env_overrides_max_actions(monkeypatch):
    monkeypatch.setenv("CAUSAL_MAX_ACTIONS", "5")
    a = _agent()
    assert a.MAX_ACTIONS == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v7 && uv run pytest tests/causal/test_agent_max_actions.py -q`
Expected: FAIL em `test_env_overrides_max_actions` (MAX_ACTIONS ignora a env).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/agent.py`, dentro de `_init_causal_state`, adicionar como
primeira linha do corpo (antes de `self._model = ...`):

```python
        self.MAX_ACTIONS = int(os.environ.get("CAUSAL_MAX_ACTIONS", type(self).MAX_ACTIONS))
```

(`os` já está importado no módulo.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v7 && uv run pytest tests/causal/test_agent_max_actions.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_max_actions.py
git commit -m "feat(causal): MAX_ACTIONS configurável por CAUSAL_MAX_ACTIONS"
```

---

### Task 2: `kaggle/build_notebook.py` — build-script do notebook

**Files:**
- Create: `kaggle/build_notebook.py`
- Test: `tests/kaggle/test_build_notebook.py` (novo), `tests/kaggle/__init__.py` (vazio, se necessário p/ import)

**Interfaces:**
- Consumes: o pacote `agents/causal/*.py` no `root` do repo.
- Produces:
  - `MODULES: list[str]` (9 nomes)
  - `read_sources(root) -> dict[str, str]` (chave = caminho relativo; valor = base64)
  - `build_notebook(sources) -> dict` (nbformat v4)
  - `main() -> None` (grava `kaggle/submission.ipynb`); helpers `_repo_root()`, `_out_path()`

- [ ] **Step 1: Write the failing test**

```python
# tests/kaggle/test_build_notebook.py
import base64
import json
import os
import importlib.util


def _load_module():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "kaggle", "build_notebook.py")
    spec = importlib.util.spec_from_file_location("build_notebook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_repo(tmp_path):
    bn = _load_module()
    pkg = tmp_path / "agents" / "causal"
    pkg.mkdir(parents=True)
    for m in bn.MODULES:
        (pkg / m).write_text(f"# {m}\nX = 1\n")
    return str(tmp_path), bn


def test_read_sources_roundtrip_base64(tmp_path):
    root, bn = _fake_repo(tmp_path)
    src = bn.read_sources(root)
    assert set(src) == {f"agents/causal/{m}" for m in bn.MODULES}
    decoded = base64.b64decode(src["agents/causal/agent.py"]).decode()
    assert decoded == "# agent.py\nX = 1\n"


def test_read_sources_includes_prior_when_present(tmp_path):
    root, bn = _fake_repo(tmp_path)
    (tmp_path / "agents" / "causal" / "prior.json").write_text('{"counts": {}}')
    assert "agents/causal/prior.json" in bn.read_sources(root)


def test_read_sources_omits_prior_when_absent(tmp_path):
    root, bn = _fake_repo(tmp_path)
    assert "agents/causal/prior.json" not in bn.read_sources(root)


def test_build_notebook_is_valid_nbformat(tmp_path):
    root, bn = _fake_repo(tmp_path)
    nb = bn.build_notebook(bn.read_sources(root))
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and len(nb["cells"]) >= 3
    json.dumps(nb)                                  # serializável
    assert all(c["cell_type"] == "code" for c in nb["cells"])


def test_cells_contain_submission_mechanics(tmp_path):
    root, bn = _fake_repo(tmp_path)
    nb = bn.build_notebook(bn.read_sources(root))
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "pip install --no-index" in text
    assert "KAGGLE_IS_COMPETITION_RERUN" in text
    assert "causalobject" in text
    assert "CAUSAL_PRIOR" in text
    assert "CAUSAL_MAX_ACTIONS" in text
    assert "CAUSAL_PRIOR_SAVE" not in text          # eval read-only
    assert "langgraph" not in text
    assert "submission.parquet" in text


def test_main_writes_valid_ipynb(tmp_path, monkeypatch):
    root, bn = _fake_repo(tmp_path)
    monkeypatch.setattr(bn, "_repo_root", lambda: root)
    monkeypatch.setattr(bn, "_out_path", lambda: str(tmp_path / "submission.ipynb"))
    bn.main()
    with open(tmp_path / "submission.ipynb") as f:
        json.load(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v7 && uv run pytest tests/kaggle/test_build_notebook.py -q`
Expected: FAIL (módulo `kaggle/build_notebook.py` inexistente).

- [ ] **Step 3: Write minimal implementation**

Criar `kaggle/build_notebook.py`:

```python
# kaggle/build_notebook.py — gera kaggle/submission.ipynb auto-contido
import base64
import json
import os

WHEELS = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
REPO = "/kaggle/working/ARC-AGI-3-Agents"
COMP_REPO = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents"

MODULES = [
    "__init__.py", "agent.py", "causal_model.py", "hud.py",
    "instrumentation.py", "novelty.py", "perception.py", "policy.py",
    "transfer.py",
]

TRIMMED_INIT = (
    "from typing import Type\n"
    "from dotenv import load_dotenv\n"
    "load_dotenv()\n"
    "from .agent import Agent, Playback\n"
    "from .swarm import Swarm\n"
    "from .causal.agent import CausalObjectAgent\n"
    'AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"causalobject": CausalObjectAgent}\n'
)

ENV = (
    "SCHEME=http\n"
    "HOST=gateway\n"
    "PORT=8001\n"
    "ARC_API_KEY=test-key-123\n"
    "ARC_BASE_URL=http://gateway:8001/\n"
    "OPERATION_MODE=online\n"
    "RECORDINGS_DIR=/kaggle/working/server_recording\n"
    "CAUSAL_PRIOR=" + REPO + "/agents/causal/prior.json\n"
    "CAUSAL_MAX_ACTIONS=100000\n"
)


def read_sources(root):
    out = {}
    for m in MODULES:
        with open(os.path.join(root, "agents", "causal", m), "rb") as f:
            out["agents/causal/" + m] = base64.b64encode(f.read()).decode()
    prior = os.path.join(root, "agents", "causal", "prior.json")
    if os.path.exists(prior):
        with open(prior, "rb") as f:
            out["agents/causal/prior.json"] = base64.b64encode(f.read()).decode()
    return out


def _cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def build_notebook(sources):
    cell0 = "!pip install --no-index --find-links %s arc-agi python-dotenv\n" % WHEELS
    cell1 = (
        "import os, shutil, base64\n"
        "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
        "    os.system(\"curl --fail --retry 999 --retry-all-errors "
        "--retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games\")\n"
        f"    shutil.copytree({COMP_REPO!r}, {REPO!r}, dirs_exist_ok=True)\n"
        f"    FILES = {json.dumps(sources)}\n"
        "    for rel, b64 in FILES.items():\n"
        f"        dst = os.path.join({REPO!r}, rel)\n"
        "        os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
        "        with open(dst, 'wb') as f:\n"
        "            f.write(base64.b64decode(b64))\n"
        f"    with open({REPO!r} + '/agents/__init__.py', 'w') as f:\n"
        f"        f.write({TRIMMED_INIT!r})\n"
        f"    with open({REPO!r} + '/.env', 'w') as f:\n"
        f"        f.write({ENV!r})\n"
        f"    os.system('cd {REPO} && MPLBACKEND=agg python main.py --agent causalobject')\n"
    )
    cell2 = (
        "import os\n"
        "if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
        "    import pandas as pd\n"
        "    pd.DataFrame([['1_0', '1', True, 1]], "
        "columns=['row_id', 'game_id', 'end_of_game', 'score'])"
        ".to_parquet('/kaggle/working/submission.parquet', index=False)\n"
    )
    return {
        "cells": [_cell(cell0), _cell(cell1), _cell(cell2)],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _out_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.ipynb")


def main():
    sources = read_sources(_repo_root())
    nb = build_notebook(sources)
    out = _out_path()
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, "with", len(sources), "embedded files")


if __name__ == "__main__":
    main()
```

Criar `tests/kaggle/__init__.py` vazio se o pytest precisar do pacote (senão omitir).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v7 && uv run pytest tests/kaggle/test_build_notebook.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Gerar o notebook e rodar a suíte inteira**

Run: `cd .claude/worktrees/causal-v7 && uv run python kaggle/build_notebook.py && python -c "import json; json.load(open('kaggle/submission.ipynb'))" && uv run pytest tests/ -q`
Expected: notebook gravado; JSON válido; PASS (todos — 109 v1–v6 + novos).

- [ ] **Step 6: Commit**

```bash
git add kaggle/build_notebook.py kaggle/submission.ipynb tests/kaggle/
git commit -m "feat(kaggle): build-script do notebook auto-contido + submission.ipynb gerado"
```

---

## Fora de escopo

- Upload do notebook no Kaggle (ação manual do usuário).
- `prior.json` (gerado pelo pré-treino ao vivo, commitado à parte).

## Pós-plano

Após o pré-treino gerar/commitar `agents/causal/prior.json`, **regerar** o
notebook (`uv run python kaggle/build_notebook.py`) para embutir o prior e
commitar o `submission.ipynb` atualizado.
