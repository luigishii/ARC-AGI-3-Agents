from agents.causal.typed_model import (
    compile_rule, verify_transition_fn, accept_rule,
    TypedWorldModel, validate_engine_contract, has_engine_contract,
    REQUIRED_ENGINE_FNS,
)

# transição de um tipo: {before, action, context, after}
_T_MOVE = [
    {"before": {"x": 1, "y": 0}, "action": "R", "context": {}, "after": {"x": 2, "y": 0}},
    {"before": {"x": 5, "y": 3}, "action": "R", "context": {}, "after": {"x": 6, "y": 3}},
]

_SRC_MOVE = (
    "def transition(obj, action, ctx):\n"
    "    o = dict(obj)\n"
    "    o['x'] = o['x'] + 1\n"
    "    return o\n"
)


# --- compile_rule ---
def test_compile_rule_valid():
    fn = compile_rule(_SRC_MOVE)
    assert callable(fn)


def test_compile_rule_syntax_error_is_none():
    assert compile_rule("def transition(o) return") is None


def test_compile_rule_missing_fn_is_none():
    assert compile_rule("def other(o, a, c):\n    return o\n") is None


# --- verify_transition_fn: replay exato POR TIPO + determinismo ---
def test_verify_accepts_correct_rule():
    fn = compile_rule(_SRC_MOVE)
    assert verify_transition_fn(fn, _T_MOVE) is True


def test_verify_rejects_on_single_mismatch():
    bad = compile_rule(
        "def transition(obj, action, ctx):\n"
        "    o = dict(obj)\n"
        "    o['x'] = o['x'] + 2\n"      # erra: prevê +2 em vez de +1
        "    return o\n"
    )
    assert verify_transition_fn(bad, _T_MOVE) is False


def test_verify_rejects_nondeterministic_rule():
    nd = compile_rule(
        "def transition(obj, action, ctx, _s=[0]):\n"
        "    _s[0] += 1\n"
        "    o = dict(obj)\n"
        "    o['x'] = o['x'] + _s[0]\n"   # estado escondido → não-determinístico
        "    return o\n"
    )
    assert verify_transition_fn(nd, _T_MOVE) is False


def test_verify_rejects_raising_rule():
    boom = compile_rule("def transition(obj, action, ctx):\n    return obj['nope']\n")
    assert verify_transition_fn(boom, _T_MOVE) is False


# --- accept_rule: compila source + verifica só as transições do tipo ---
def test_accept_rule_from_source_correct():
    assert accept_rule(_SRC_MOVE, _T_MOVE) is True


def test_accept_rule_from_source_wrong():
    wrong = "def transition(obj, action, ctx):\n    return dict(obj)\n"  # não move
    assert accept_rule(wrong, _T_MOVE) is False


def test_accept_rule_non_compiling():
    assert accept_rule("def transition(o) return", _T_MOVE) is False


# --- TypedWorldModel: aplica f_τ por objeto, monta próximo estado ---
def test_typed_model_predicts_per_type():
    m = TypedWorldModel()
    m.set_rule("mover", _SRC_MOVE)
    objs = [("mover", {"x": 1, "y": 0}), ("mover", {"x": 9, "y": 9})]
    nxt = m.predict(objs, "R")
    assert nxt == [("mover", {"x": 2, "y": 0}), ("mover", {"x": 10, "y": 9})]


def test_typed_model_unknown_type_unchanged():
    m = TypedWorldModel()
    objs = [("wall", {"x": 3, "y": 3})]
    assert m.predict(objs, "R") == [("wall", {"x": 3, "y": 3})]


def test_typed_model_serialization_roundtrip():
    m = TypedWorldModel()
    m.set_rule("mover", _SRC_MOVE)
    m2 = TypedWorldModel.from_dict(m.to_dict())
    objs = [("mover", {"x": 1, "y": 0})]
    assert m2.predict(objs, "R") == m.predict(objs, "R")


# --- contrato das 4 funções (OPINE game_engine.py) ---
_ENGINE_SRC = (
    "def transition_function(state, action):\n    return state\n"
    "def reward_function(state):\n    return (0.0, False)\n"
    "def extract_objects(frame):\n    return []\n"
    "def planner():\n    return None\n"
)


def test_required_engine_fns():
    assert REQUIRED_ENGINE_FNS == ("transition_function", "reward_function", "extract_objects")


def test_validate_engine_contract_all_present():
    v = validate_engine_contract(_ENGINE_SRC)
    assert v["transition_function"] and v["reward_function"] and v["extract_objects"]
    assert v["planner"] is True          # opcional presente aqui
    assert has_engine_contract(_ENGINE_SRC) is True


def test_validate_engine_contract_missing_required():
    src = "def transition_function(state, action):\n    return state\n"
    assert has_engine_contract(src) is False
    v = validate_engine_contract(src)
    assert v["transition_function"] is True
    assert v["reward_function"] is False
