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
    WHEELS, REPO, COMP_REPO, TRIMMED_INIT, MODEL_DATASET_PATH, MODEL_DISCOVERY,
    KERNELS_WHEELS, HF_CACHE_HOME,
    read_sources, _cell, _repo_root,
)

# .env offline: sem HTTP, sem ARC_API_KEY (offline nao chama API). ENVIRONMENTS_DIR
# e completado em runtime pela Cell 1 (descoberto por glob de metadata.json).
OFFLINE_ENV = (
    "DEBUG=True\n"
    "OPERATION_MODE=offline\n"
    "RECORDINGS_DIR=/kaggle/working/recordings\n"
    "CAUSAL_PRIOR=" + REPO + "/agents/causal/prior.json\n"
    # CAUSAL_MAX_ACTIONS: NAO setado — budget dinamico por jogo (agent.py _GAME_BUDGETS).
    # Click-only=80, navegacao=150, complexo=200. Early-exit a 65% sem level_up.
    "CAUSAL_LLM=1\n"
    "CAUSAL_LLM_MAX_CALLS=3\n"     # max 3 chamadas LLM por jogo (~90s max)
    "CAUSAL_LLM_DEFER=50\n"        # 50 acoes heuristicas antes de LLM (adaptive: reduz se preso)
    "CAUSAL_TYPED=1\n"
    "CAUSAL_ETA=1\n"
    "CAUSAL_IW=1\n"
    "CAUSAL_RPROG=1\n"       # progresso model-free por reward real (Lever B')
    "CAUSAL_COVER=1\n"
    "CAUSAL_CLICKMAP=1\n"
    "CAUSAL_GROUNDED=1\n"       # reward grounded (manhattan, multi-align, pattern)
    "CAUSAL_FIX=1\n"        # guarda global anti-fixacao
    "CAUSAL_DIRECT=1\n"     # score-max Lever #2: raciocinio direto passo-a-passo
    "CAUSAL_DIRECT_COOLDOWN=20\n"  # cooldown entre chamadas direct (default 2 -> 20)
    "CAUSAL_CLASS=1\n"      # 1 chamada/jogo: LLM classifica o jogo (A-F) + papeis
    "SWARM_GAME_TIMEOUT=600\n"  # gpt-oss ~30s/chamada: 120s estourava no meio e sobrepunha jogos
    "CAUSAL_EFFORT=medium\n"  # gpt-oss: esforco de raciocinio (low|medium|high) no Harmony
    "HF_HUB_OFFLINE=1\n"       # kernels/hub sem rede: le so o cache local
    "TRANSFORMERS_OFFLINE=1\n"
    "HF_HOME=" + HF_CACHE_HOME + "\n"   # cache com o repo kernels-community/triton_kernels
    "QWEN_MODEL_PATH=" + MODEL_DATASET_PATH + "\n"
)

# Gargalo #2 RESOLVIDO p/ de-blinding: em vez de 1 subprocesso por jogo (recarrega o 32B
# 61GB a cada troca e TRAVA), roda UMA chamada de main.py sem --game → o modelo é singleton
# por processo, carrega 1x e joga o subconjunto OFFLINE_GAMES no MESMO processo.
OFFLINE_GAMES = ""   # "" = TODOS os 25; ex: "vc33,ls20" (prefixos)
# "1" = usa a tabela _GAME_KNOWLEDGE (mecanica dos 25 publicos). "0" = MODO CEGO: ignora a
# tabela (so heuristicas genericas + classe inferida pelo LLM) = o que a eval privada ve.
CAUSAL_GK = "1"


def build_offline_notebook(sources):
    # Embarca o main.py corrigido (offline: lista jogos via Arcade local em vez de
    # HTTP). O copytree traz o main.py da competicao; nossos FILES o sobrescrevem.
    # agents/agent.py ja vem via read_sources().
    with open(os.path.join(_repo_root(), "main.py"), "rb") as f:
        sources = {**sources, "main.py": base64.b64encode(f.read()).decode()}
    cell0 = (
        "!pip install --no-index --find-links %s arc-agi python-dotenv\n" % WHEELS
        + "import glob as _g, os as _o, shutil as _sh, subprocess as _sp\n"
        "_wh = _g.glob('/kaggle/input/**/kernels-*.whl', recursive=True)\n"
        "_wd = _o.path.dirname(_wh[0]) if _wh else ''\n"
        "print('KERNELS_WHEELS =', _wd or 'NAO ACHOU')\n"
        "if _wd:\n"
        "    print(_sp.run(['pip','install','--no-index','--find-links',_wd,'kernels==0.14.0',"
        "'huggingface_hub'], capture_output=True, text=True).stderr[-400:] or 'kernels OK')\n"
        "_hc = _g.glob('/kaggle/input/**/hub/models--kernels-community--triton_kernels', "
        "recursive=True)\n"
        "if _hc:\n"
        "    _src = _o.path.dirname(_o.path.dirname(_hc[0]))\n"   # dir que contem hub/
        "    _sh.copytree(_src, '/kaggle/working/hfcache', dirs_exist_ok=True)\n"
        "    print('HF_HOME writable -> /kaggle/working/hfcache')\n"
        "else:\n"
        "    print('AVISO: hub cache de kernels nao encontrado')\n")
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
        "_tk = glob.glob('/kaggle/input/**/build/torch-universal/triton_kernels/__init__.py', "
        "recursive=True)\n"
        "if _tk:\n"
        "    _tu = os.path.dirname(os.path.dirname(_tk[0]))\n"          # .../build/torch-universal
        "    _reporoot = os.path.dirname(os.path.dirname(_tu))\n"       # dir que contem build/
        "    shutil.copytree(_reporoot, '/kaggle/working/tkrepo', dirs_exist_ok=True)\n"
        "    _lk = '/kaggle/working/tkrepo/build/torch-universal'\n"
        "else:\n"
        "    _lk = ''\n"
        "print('GPT_OSS_KERNEL_DIR ->', _lk or 'NAO ACHOU triton_kernels/ (gpt-oss falha)')\n"
        f"with open({REPO!r} + '/.env', 'w') as f:\n"
        f"    f.write({OFFLINE_ENV!r} + 'ENVIRONMENTS_DIR=' + env_dir + '\\n'\n"
        "            + 'HF_HOME=/kaggle/working/hfcache\\n'\n"
        "            + ('GPT_OSS_KERNEL_DIR=' + _lk + '\\n' if _lk else ''))\n"   # override: cache gravavel
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
        "# >>> EDITE AQUI: '1' = tabela de mecanicas dos 25 publicos; '0' = MODO CEGO"
        " (= eval privada) <<<\n"
        f"CAUSAL_GK = {CAUSAL_GK!r}\n"
        + MODEL_DISCOVERY
        + f"with open({REPO!r} + '/.env', 'a') as f:\n"
        "    f.write('OFFLINE_GAMES=' + OFFLINE_GAMES + '\\n')\n"
        "    f.write('CAUSAL_GK=' + CAUSAL_GK + '\\n')\n"
        "    f.write('QWEN_MODEL_PATH=' + _mp + '\\n')\n"
        "print('rodando OFFLINE_GAMES=%r (1 processo, 32B 1x)' % OFFLINE_GAMES)\n"
        f"os.system('cd {REPO} && MPLBACKEND=agg python main.py --agent causalobject')\n"
    )
    cell2 = (
        "import glob, json\n"
        "# Diagnostico do Swarm: resumo por-jogo (sequencial + priorizado)\n"
        "for p in sorted(glob.glob('/kaggle/working/**/swarm_diagnostics.json', recursive=True)):\n"
        "    print('=== SWARM DIAGNOSTICS ===')\n"
        "    d = json.loads(open(p).read())\n"
        "    print(f\"Total: {d['total_levels']}L em {d['total_time_s']}s ({d['total_games']} jogos)\")\n"
        "    for g in d.get('games', []):\n"
        "        print(f\"  {g['game_id']:6s} | {g['levels']}L | {g['actions']:3d} acts | \"\n"
        "              f\"{g['time_s']:6.1f}s | {g['fps']:6.2f} fps\")\n"
        "# Diagnostico causal por-agente\n"
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
