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


def test_notebook_packages_llm_and_env(tmp_path):
    root, bn = _fake_repo(tmp_path)
    assert "llm.py" in bn.MODULES and "planning.py" in bn.MODULES
    nb = bn.build_notebook(bn.read_sources(root))
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "agents/causal/llm.py" in text        # módulo embutido
    assert "CAUSAL_LLM" in text
    assert "QWEN_MODEL_PATH" in text


def test_main_writes_valid_ipynb(tmp_path, monkeypatch):
    root, bn = _fake_repo(tmp_path)
    monkeypatch.setattr(bn, "_repo_root", lambda: root)
    monkeypatch.setattr(bn, "_out_path", lambda: str(tmp_path / "submission.ipynb"))
    bn.main()
    with open(tmp_path / "submission.ipynb") as f:
        json.load(f)


def test_env_has_rprog():
    import kaggle.build_notebook as b
    assert "CAUSAL_RPROG=1" in b.ENV


def test_env_has_cover():
    import kaggle.build_notebook as b
    assert "CAUSAL_COVER=1" in b.ENV
