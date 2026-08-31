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
