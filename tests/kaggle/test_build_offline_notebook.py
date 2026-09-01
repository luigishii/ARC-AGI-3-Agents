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


def test_patched_mainpy_embedded():
    # O main.py corrigido (lista jogos offline via Arcade) precisa ser embarcado,
    # senao o copytree usa o main.py da competicao que so busca jogos via HTTP.
    _, run = _run_cell()
    assert '"main.py"' in run


def test_offline_env_has_rprog():
    import kaggle.build_offline_notebook as b
    assert "CAUSAL_RPROG=1" in b.OFFLINE_ENV


def test_offline_runs_subset_in_one_process():
    import kaggle.build_offline_notebook as b
    assert isinstance(b.OFFLINE_GAMES, str)              # subconjunto configurável
    src = b.read_sources(b._repo_root())
    nb = b.build_offline_notebook(src)
    text = "".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"]
    )
    assert "OFFLINE_GAMES" in text                        # var editável na célula + .env
    # UMA chamada de main.py sem --game (modelo 1x, joga o subconjunto in-process)
    assert "python main.py --agent causalobject'" in text
    assert "for g in sel:" not in text                   # sem loop por-jogo (reload)


def test_offline_env_has_cover():
    import kaggle.build_offline_notebook as b
    assert "CAUSAL_COVER=1" in b.OFFLINE_ENV


def test_offline_env_has_fix():
    import kaggle.build_offline_notebook as b
    assert "CAUSAL_FIX=1" in b.OFFLINE_ENV


def test_offline_env_has_direct():
    import kaggle.build_offline_notebook as b
    assert "CAUSAL_DIRECT=1" in b.OFFLINE_ENV
