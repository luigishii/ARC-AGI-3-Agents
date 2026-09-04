# kaggle/build_notebook.py — gera kaggle/submission.ipynb auto-contido
import base64
import json
import os

WHEELS = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
REPO = "/kaggle/working/ARC-AGI-3-Agents"
COMP_REPO = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents"

MODULES = [
    "__init__.py", "agent.py", "causal_model.py", "hud.py",
    "instrumentation.py", "novelty.py", "perception.py", "perception_strategy.py", "policy.py",
    "transfer.py", "planning.py", "navigate.py", "llm.py", "sandbox.py", "dsl.py", "ranker.py",
    "ontology.py", "typed_model.py", "iw.py", "goals.py",
]

# Caminho do dataset de pesos do LLM no Kaggle. O usuário anexa o dataset ao
# notebook e edita este valor para o slug do seu dataset, depois regenera o
# notebook (uv run python kaggle/build_notebook.py).
# >>> EDITE AQUI <<< com o slug real do gpt-oss-120b anexado (catalogo Kaggle Models:
# danielhanchen/gpt-oss-120b). O path DEVE conter "gpt-oss" -> ativa o modo Harmony
# automaticamente (llm._should_use_harmony): chat template de raciocinio + canal final.
MODEL_DATASET_PATH = "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1"

# Descoberta do path REAL dos pesos em runtime (o slug do Kaggle Models varia por
# owner/variation/version: ex. openai/gpt-oss-120b/transformers/gpt-oss-120b/1).
# Se MODEL_DATASET_PATH nao existe, procura config.json com "gpt-oss" no path e usa o
# mais raso. Sem isso o HFClient falha e o run cai SILENCIOSAMENTE pro NullLLMClient.
# A linha QWEN_MODEL_PATH gravada por ultimo no .env VENCE a do ENV (dotenv: last wins).
MODEL_DISCOVERY = (
    "import glob as _mg, os as _mo\n"
    f"_mp = {MODEL_DATASET_PATH!r}\n"
    "if not _mo.path.isdir(_mp):\n"
    "    _cf = [p for p in _mg.glob('/kaggle/input/models/**/config.json', recursive=True)\n"
    "           if 'gpt-oss' in p.lower()]\n"
    "    _cf += [p for p in _mg.glob('/kaggle/input/**/config.json', recursive=True)\n"
    "            if 'gpt-oss' in p.lower() and p not in _cf]\n"
    "    if _cf:\n"
    "        _mp = _mo.path.dirname(sorted(_cf, key=lambda p: (p.count('/'), p))[0])\n"
    "print('QWEN_MODEL_PATH ->', _mp, '(OK)' if _mo.path.isdir(_mp) else "
    "'(NAO EXISTE: LLM vai cair pro NullLLMClient)')\n"
)

# gpt-oss MXFP4: os kernels vem da lib HF `kernels` + o repo kernels-community/triton_kernels
# (torch-universal, ~536KB). Air-gapped: com internet ON o usuario faz
#   pip download kernels huggingface_hub -d ./offline_wheels
#   HF_HOME=./hf_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('kernels-community/triton_kernels')"
# e sobe ./offline_wheels e ./hf_cache como 2 datasets. >>> EDITE AQUI <<< com os slugs reais.
KERNELS_WHEELS = "/kaggle/input/gpt-oss-offline-kernels/offline_wheels"  # wheels kernels+hub
HF_CACHE_HOME = "/kaggle/input/gpt-oss-offline-kernels/hf_cache"         # HF_HOME (contem hub/)

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
    "CAUSAL_MAX_ACTIONS=1500\n"    # reativa early-exit (65% sem level-up) na eval; era 100000
    "CAUSAL_LLM=1\n"
    # gpt-oss ~30s/chamada + Swarm PARALELO serializado por lock -> sem estes caps a
    # submissao inteira fica na fila do LLM (a de 0.08 usava Qwen3-32B a 1.1s).
    "CAUSAL_LLM_MAX_CALLS=4\n"      # por jogo
    "CAUSAL_LLM_DEFER=50\n"         # acoes heuristicas antes do LLM
    "CAUSAL_DIRECT_COOLDOWN=20\n"   # direct esparso (default 2)
    "CAUSAL_LLM_TOTAL_CALLS=600\n"  # cap global por processo (~5h a 30s/chamada)
    "CAUSAL_DIRECT_EFFORT=low\n"    # direct em low (rapido); classe/reward em CAUSAL_EFFORT
    "SWARM_DEADLINE_S=30000\n"      # 8h20 desde o inicio do Swarm: fecha o scorecard antes do kill
    "CAUSAL_TYPED=1\n"       # síntese fatorada f_τ (Qwen valida por-tipo via accept_rule)
    "CAUSAL_ETA=1\n"         # exploração por erro de ontologia (η)
    "CAUSAL_IW=1\n"          # planner Iterated Width sobre o TypedWorldModel
    "CAUSAL_RPROG=1\n"       # progresso model-free por reward real (Lever B')
    "CAUSAL_COVER=1\n"
    "CAUSAL_CLICKMAP=1\n"
    "CAUSAL_GROUNDED=1\n"       # exploração por cobertura + anti-fixação
    "CAUSAL_FIX=1\n"        # guarda global anti-fixação
    "CAUSAL_DIRECT=1\n"     # score-max Lever #2: raciocinio direto passo-a-passo
    "CAUSAL_CLASS=1\n"      # 1 chamada/jogo: LLM classifica o jogo (A-F) + papeis (jogo nao-visto)
    "CAUSAL_EFFORT=medium\n"  # gpt-oss: esforco de raciocinio (low|medium|high) no Harmony
    "HF_HUB_OFFLINE=1\n"       # kernels/hub sem rede: le so o cache local
    "TRANSFORMERS_OFFLINE=1\n"
    "HF_HOME=" + HF_CACHE_HOME + "\n"   # cache com o repo kernels-community/triton_kernels
    "QWEN_MODEL_PATH=" + MODEL_DATASET_PATH + "\n"
)


def read_sources(root):
    out = {}
    for m in MODULES:
        with open(os.path.join(root, "agents", "causal", m), "rb") as f:
            out["agents/causal/" + m] = base64.b64encode(f.read()).decode()
    # agents/agent.py (base class) tem logs melhorados — embarcar tambem
    with open(os.path.join(root, "agents", "agent.py"), "rb") as f:
        out["agents/agent.py"] = base64.b64encode(f.read()).decode()
    # agents/swarm.py: sequencial + priorizacao + diagnosticos
    with open(os.path.join(root, "agents", "swarm.py"), "rb") as f:
        out["agents/swarm.py"] = base64.b64encode(f.read()).decode()
    prior = os.path.join(root, "agents", "causal", "prior.json")
    if os.path.exists(prior):
        with open(prior, "rb") as f:
            out["agents/causal/prior.json"] = base64.b64encode(f.read()).decode()
    return out


def _cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def build_notebook(sources):
    cell0 = (
        "!pip install --no-index --find-links %s arc-agi python-dotenv\n" % WHEELS
        + "!pip install --no-index --find-links %s kernels huggingface_hub || "
        "echo 'kernels wheels ausentes -> gpt-oss cairia em bf16'\n" % KERNELS_WHEELS)
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
        + "".join("    " + ln + "\n" for ln in MODEL_DISCOVERY.splitlines())
        + f"    with open({REPO!r} + '/.env', 'a') as f:\n"
        "        f.write('QWEN_MODEL_PATH=' + _mp + '\\n')\n"
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
