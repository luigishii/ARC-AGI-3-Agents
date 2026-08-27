from agents.causal.sandbox import compile_decide, run_decide, execute_code_goal

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
