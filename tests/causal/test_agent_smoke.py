from arcengine import FrameData, GameAction, GameState

from agents.causal.agent import CausalObjectAgent


def _agent():
    # Instanciar sem tocar rede: bypass __init__ do harness.
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.frames = []
    a._init_causal_state()
    return a


def test_reset_when_not_played():
    a = _agent()
    f = FrameData(levels_completed=0, state=GameState.NOT_PLAYED)
    assert a.choose_action([f], f) is GameAction.RESET


def test_returns_available_action_when_playing():
    a = _agent()
    f = FrameData(
        levels_completed=0,
        state=GameState.NOT_FINISHED,
        frame=[[[0] * 64 for _ in range(64)]],
        available_actions=[GameAction.ACTION1],
    )
    act = a.choose_action([f], f)
    assert act in (GameAction.ACTION1,)
