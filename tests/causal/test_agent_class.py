from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent
from agents.causal.hud import HudMask

_CLASS_JSON = ('{"cls":"A","avatar":9,"target":5,"click":[9,1],'
               '"hud_rows":[63],"hud_cols":[]}')


class _Fake:
    def __init__(self, canned):
        self.canned = canned
        self.calls = 0
        self.prompts = []

    def complete(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        return self.canned


class _Frame:
    def __init__(self, frame, available=None, levels=0):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = levels
        self.available_actions = available or [GameAction.ACTION1]
        self.full_reset = False


def _grid():
    g = [[0] * 64 for _ in range(64)]
    g[10][10] = 9
    return [g]


def _agent(monkeypatch, game_id="zzzz-unknown", counter=8, **env):
    env.setdefault("CAUSAL_LLM", "1")
    env.setdefault("CAUSAL_LLM_DEFER", "50")     # class infer NAO espera o defer
    env.setdefault("CAUSAL_LLM_MAX_CALLS", "3")
    env.setdefault("CAUSAL_MAX_ACTIONS", "10000")
    env.setdefault("CAUSAL_DIRECT", "0")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = counter
    a.game_id = game_id
    a.MAX_ACTIONS = 10000
    a._init_causal_state()
    return a


def test_class_infer_fills_gk_on_unknown_game(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_CLASS="1")
    a._llm = _Fake(_CLASS_JSON)
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1, GameAction.ACTION6]))
    assert a._llm.calls == 1
    assert a._gk["cls"] == "A" and a._gk["avatar"] == 9 and a._gk["click"] == {9, 1}
    assert a._hud.row_count[63] == HudMask._SEED       # HUD re-seedado
    assert a.phase2_stats()["gk_src"] == "llm:A"
    assert "sokoban" in a._llm.prompts[0].lower()      # usou o prompt de classe


def test_class_infer_waits_for_min_actions(monkeypatch):
    a = _agent(monkeypatch, counter=2, CAUSAL_CLASS="1")
    a._llm = _Fake(_CLASS_JSON)
    a.choose_action([], _Frame(_grid()))
    assert a._llm.calls == 0


def test_class_infer_skipped_when_table_knows_game(monkeypatch):
    a = _agent(monkeypatch, game_id="vc33-x", CAUSAL_CLASS="1")
    a._llm = _Fake(_CLASS_JSON)
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION6]))
    assert a._llm.calls == 0
    assert a._gk["cls"] == "C"


def test_class_infer_off_by_default(monkeypatch):
    monkeypatch.delenv("CAUSAL_CLASS", raising=False)
    a = _agent(monkeypatch)
    a._llm = _Fake(_CLASS_JSON)
    a.choose_action([], _Frame(_grid()))
    assert a._llm.calls == 0


def test_class_infer_invalid_reply_is_not_retried(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_CLASS="1")
    a._llm = _Fake("garbage")
    a.choose_action([], _Frame(_grid()))
    a.action_counter += 1
    a.choose_action([], _Frame(_grid()))
    assert a._llm.calls == 1
    assert a._gk == {}
    assert a.phase2_stats()["gk_src"] is None


def test_class_infer_runs_in_blind_mode(monkeypatch):
    a = _agent(monkeypatch, game_id="vc33-x", CAUSAL_CLASS="1", CAUSAL_GK="0")
    a._llm = _Fake(_CLASS_JSON)
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION6]))
    assert a._llm.calls == 1 and a._gk["cls"] == "A"
