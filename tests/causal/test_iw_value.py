from agents.causal.iw import iw_search, iw_plan
from agents.causal.typed_model import TypedWorldModel


# regra que ramifica na ação: A → x+1, qualquer outra → x-1
_BRANCH = (
    "def transition(obj, action, ctx):\n"
    "    o = dict(obj)\n"
    "    o['x'] = o['x'] + 1 if action == 'A' else o['x'] - 1\n"
    "    return o\n"
)
# regra monotônica: sempre x+1 (qualquer ação)
_UP = "def transition(obj, action, ctx):\n    o = dict(obj)\n    o['x'] = o['x'] + 1\n    return o\n"


def _model(rule):
    m = TypedWorldModel()
    m.set_rule("t", rule)
    return m


def _start(x=0):
    return [("t", {"x": x, "y": 0, "color": 3})]


# --- best-first escolhe a ação que sobe o valor ---
def test_value_picks_improving_action():
    higher_x = lambda st: float(st[0][1]["x"])
    out = iw_search(_start(0), ["A", "B"], _model(_BRANCH),
                    value_fn=higher_x, width=1)
    assert out == "A"


# --- nada melhora o start → None ---
def test_value_none_when_no_improvement():
    lower_x = lambda st: -float(st[0][1]["x"])   # premia x menor; _UP só aumenta x
    out = iw_search(_start(0), ["A"], _model(_UP),
                    value_fn=lower_x, width=1, max_nodes=50)
    assert out is None


# --- goal_fn tem precedência sobre value_fn (comportamento atual inalterado) ---
def test_goal_fn_takes_precedence_over_value():
    goal = lambda st: st[0][1]["x"] >= 2
    ignored_value = lambda st: -999.0
    out = iw_search(_start(0), ["A"], _model(_UP),
                    goal_fn=goal, value_fn=ignored_value, width=1)
    assert out == "A"


# --- exploração pura (sem goal_fn nem value_fn) inalterada ---
def test_pure_exploration_unchanged():
    out = iw_search(_start(0), ["A"], _model(_UP), width=1)
    assert out == "A"


# --- iw_plan repassa value_fn e escala a largura ---
def test_iw_plan_value_best_first():
    higher_x = lambda st: float(st[0][1]["x"])
    out = iw_plan(_start(0), ["A", "B"], _model(_BRANCH),
                  value_fn=higher_x, max_width=2)
    assert out == "A"
