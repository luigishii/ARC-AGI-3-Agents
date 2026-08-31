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
