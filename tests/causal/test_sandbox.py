from agents.causal.sandbox import (
    compile_decide, run_decide, execute_code_goal,
    classify_error, run_decide_diag, execute_code_goal_diag,
    execute_code_goal_verified,
)

_SCENE = object()   # o sandbox só passa scene ao decide; conteúdo é opaco aqui


def test_compile_valid():
    fn = compile_decide("def decide(scene):\n    return 'ACTION1'\n")
    assert callable(fn)


def test_compile_syntax_error_is_none():
    assert compile_decide("def decide(scene) return") is None


def test_compile_without_decide_is_none():
    assert compile_decide("x = 1") is None


def test_run_returns_action_key():
    fn = compile_decide("def decide(scene):\n    return 'ACTION6@cell=2,3'\n")
    assert run_decide(fn, _SCENE) == "ACTION6@cell=2,3"


def test_run_exception_is_none():
    fn = compile_decide("def decide(scene):\n    return 1/0\n")
    assert run_decide(fn, _SCENE) is None


def test_run_non_string_is_none():
    fn = compile_decide("def decide(scene):\n    return 42\n")
    assert run_decide(fn, _SCENE) is None


def test_import_is_blocked():
    src = "def decide(scene):\n    import os\n    return os.getcwd()\n"
    assert execute_code_goal(src, _SCENE) is None


def test_infinite_loop_times_out():
    src = "def decide(scene):\n    while True:\n        pass\n"
    assert execute_code_goal(src, _SCENE, timeout=0.1) is None


def test_execute_code_goal_end_to_end():
    src = "def decide(scene):\n    return 'ACTION2'\n"
    assert execute_code_goal(src, _SCENE) == "ACTION2"


# --- §3: taxonomia de erros por regex ---
def test_classify_index_oob():
    try:
        [][3]
    except Exception as e:  # noqa: BLE001
        assert classify_error(e) == "index_oob"


def test_classify_syntax_error_text():
    assert classify_error("SyntaxError: invalid syntax") == "syntax_error"


def test_classify_shape_mismatch_text():
    assert classify_error("ValueError: operands could not be broadcast together") \
        == "shape_mismatch"


def test_classify_semantic_error_is_other():
    try:
        1 / 0
    except Exception as e:  # noqa: BLE001
        assert classify_error(e) == "other"


def test_classify_none():
    assert classify_error(None) == "none"


# --- §3: run_decide_diag ---
def test_diag_success():
    fn = compile_decide("def decide(scene):\n    return 'ACTION1'\n")
    assert run_decide_diag(fn, _SCENE) == ("ACTION1", None)


def test_diag_exception_slug():
    fn = compile_decide("def decide(scene):\n    return [][5]\n")
    assert run_decide_diag(fn, _SCENE) == (None, "index_oob")


def test_diag_no_op_on_non_string():
    fn = compile_decide("def decide(scene):\n    return 42\n")
    assert run_decide_diag(fn, _SCENE) == (None, "no_op")


def test_diag_timeout_is_infinite_loop():
    fn = compile_decide("def decide(scene):\n    while True:\n        pass\n")
    assert run_decide_diag(fn, _SCENE, timeout=0.1) == (None, "infinite_loop")


def test_execute_code_goal_diag_syntax():
    assert execute_code_goal_diag("def decide(scene) return", _SCENE) \
        == (None, "syntax_error")


# --- OPINE double-eval: determinismo ---
def test_verified_accepts_deterministic():
    src = "def decide(scene):\n    return 'ACTION1'\n"
    assert execute_code_goal_verified(src, _SCENE) == "ACTION1"


def test_verified_rejects_nondeterministic():
    # estado escondido em default mutável → alterna a saída entre chamadas
    src = ("def decide(scene, _s=[0]):\n"
           "    _s[0] += 1\n"
           "    return 'ACTION1' if _s[0] % 2 else 'ACTION2'\n")
    assert execute_code_goal_verified(src, _SCENE) is None
