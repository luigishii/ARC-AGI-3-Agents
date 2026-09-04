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


class _Seq:
    """FakeLLM que devolve respostas em sequencia e grava prompt+effort."""
    def __init__(self, *canned):
        self.canned, self.calls, self.prompts, self.efforts = list(canned), 0, [], []

    def complete(self, prompt, effort=None):
        self.calls += 1
        self.prompts.append(prompt)
        self.efforts.append(effort)
        return self.canned[min(self.calls - 1, len(self.canned) - 1)]


def test_inferred_class_reaches_direct_prompt_and_efforts(monkeypatch):
    monkeypatch.delenv("CAUSAL_DIRECT_EFFORT", raising=False)
    monkeypatch.setenv("CAUSAL_EFFORT", "medium")
    a = _agent(monkeypatch, CAUSAL_CLASS="1", CAUSAL_DIRECT="1", CAUSAL_LLM_DEFER="0",
               CAUSAL_DIRECT_COOLDOWN="0")
    a._llm = _Seq(_CLASS_JSON, '{"type":"press","action":"ACTION1"}')
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1, GameAction.ACTION6]))
    assert a._llm.calls == 2
    assert "GAME CLASS: A" in a._llm.prompts[1]
    assert "AVATAR COLOR: 9" in a._llm.prompts[1]
    assert a._llm.efforts == ["medium", "low"]     # classe=medium, direct=low


def _blind_budget_agent(monkeypatch, **env):
    monkeypatch.delenv("CAUSAL_MAX_ACTIONS", raising=False)
    env.setdefault("CAUSAL_LLM", "1")
    env.setdefault("CAUSAL_CLASS", "1")
    env.setdefault("CAUSAL_DIRECT", "0")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 8
    a.game_id = "zzzz-unknown"
    a._init_causal_state()
    return a


def test_budget_follows_inferred_class_keyboard(monkeypatch):
    a = _blind_budget_agent(monkeypatch)
    a._llm = _Fake(_CLASS_JSON)                     # cls A (navegacao)
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1, GameAction.ACTION2]))
    assert a.MAX_ACTIONS == 150


def test_budget_follows_inferred_class_click_only(monkeypatch):
    a = _blind_budget_agent(monkeypatch)
    a._llm = _Fake('{"cls":"C","avatar":null,"target":null,"click":[9],"hud_rows":[],"hud_cols":[]}')
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION6]))
    assert a.MAX_ACTIONS == 80


def test_budget_env_override_wins_over_class(monkeypatch):
    a = _blind_budget_agent(monkeypatch, CAUSAL_MAX_ACTIONS="1500")
    a._llm = _Fake(_CLASS_JSON)
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert a.MAX_ACTIONS == 1500
