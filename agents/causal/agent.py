from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


class CausalObjectAgent(Agent):
    """Agente objeto-cêntrico causal (v1). Ver docs/superpowers/specs/2026-08-27-causal-object-agent-design.md."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_causal_state()

    def _init_causal_state(self) -> None:
        # Preenchido nas tasks seguintes (perception/model/policy/instrumentation).
        self._prev_scene = None
        self._last_action = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or getattr(
            latest_frame, "full_reset", False
        ):
            return GameAction.RESET
        raw = latest_frame.available_actions or [GameAction.ACTION1.value]
        first = raw[0]
        action = first if isinstance(first, GameAction) else GameAction.from_id(first)
        action.reasoning = {"stage": "stub"}
        return action
