# kaggle/build_offline_notebook.py — gera kaggle/offline.ipynb (harness OFFLINE de
# de-blinding). Roda o agente completo (Qwen3-32B + flags) contra o Arcade OFFLINE
# no EDITOR do Kaggle (RTX Pro 6000, internet OFF), com logs visiveis. Reusa as pecas
# do build_notebook.py (DRY): so muda o .env (offline, sem chave) e as cells de run.
import base64
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
    "CAUSAL_RPROG=1\n"       # progresso model-free por reward real (Lever B')
    "CAUSAL_COVER=1\n"       # exploração por cobertura + anti-fixação
    "QWEN_MODEL_PATH=" + MODEL_DATASET_PATH + "\n"
)

# Gargalo #2 RESOLVIDO p/ de-blinding: em vez de 1 subprocesso por jogo (recarrega o 32B
# 61GB a cada troca e TRAVA), roda UMA chamada de main.py sem --game → o modelo é singleton
# por processo, carrega 1x e joga o subconjunto OFFLINE_GAMES no MESMO processo.
OFFLINE_GAMES = "sk48,vc33,ls20,tn36"   # "" = TODOS os 25; ex: "vc33,ls20" (prefixos)


def build_offline_notebook(sources):
    # Embarca o main.py corrigido (offline: lista jogos via Arcade local em vez de
    # HTTP). O copytree traz o main.py da competicao; nossos FILES o sobrescrevem.
    with open(os.path.join(_repo_root(), "main.py"), "rb") as f:
        sources = {**sources, "main.py": base64.b64encode(f.read()).decode()}
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
        "# >>> EDITE AQUI: quais jogos rodar. '' = TODOS os 25. Ex: 'vc33,ls20,tn36' <<<\n"
        "# O 32B carrega UMA vez e joga o subconjunto no MESMO processo (sem reload/trava).\n"
        f"OFFLINE_GAMES = {OFFLINE_GAMES!r}\n"
        f"with open({REPO!r} + '/.env', 'a') as f:\n"
        "    f.write('OFFLINE_GAMES=' + OFFLINE_GAMES + '\\n')\n"
        "print('rodando OFFLINE_GAMES=%r (1 processo, 32B 1x)' % OFFLINE_GAMES)\n"
        f"os.system('cd {REPO} && MPLBACKEND=agg python main.py --agent causalobject')\n"
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
