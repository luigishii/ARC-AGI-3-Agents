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
