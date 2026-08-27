from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from .causal_model import CausalModel
from .instrumentation import Instrumentation
from .perception import match_objects, parse
from .policy import Policy


class CausalObjectAgent(Agent):
    """Agente objeto-cêntrico causal (v1). Ver docs/superpowers/specs/2026-08-27-causal-object-agent-design.md."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_causal_state()

    def _init_causal_state(self) -> None:
        self._model = CausalModel()
        self._policy = Policy(seed=0, epsilon=0.05)
        self._instr = Instrumentation()
        self._prev_scene = None
        self._last_key = None
        self._last_predicted = None
        self._last_level = 0
        self._seen_effects = set()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or getattr(
            latest_frame, "full_reset", False
        ):
            self._prev_scene = None
            self._last_key = None
            return GameAction.RESET

        scene = match_objects(self._prev_scene, parse(latest_frame.frame))

        # fecha o loop causal da ação anterior
        if self._prev_scene is not None and self._last_key is not None:
            level_up = (latest_frame.levels_completed or 0) > self._last_level
            actual = self._model.observe(self._prev_scene, self._last_key, scene, level_up)
            self._model.record_prediction(self._last_predicted, actual)
            self._seen_effects.add(actual.kind)

        # decide a próxima ação
        budget_frac = 1.0
        if self.MAX_ACTIONS not in (0, float("inf")):
            budget_frac = max(0.0, 1 - self.action_counter / self.MAX_ACTIONS)
        cand = self._policy.decide(
            scene, self._model, latest_frame.available_actions or [GameAction.ACTION1],
            self._seen_effects, budget_frac,
        )
        action = cand.action
        if action.is_complex():
            action.set_data({"x": cand.x, "y": cand.y})

        predicted, conf = self._model.predict(cand.key)
        mode = "EXPLOIT" if self._model.is_progress(cand.key) else "EXPLORE"
        action.reasoning = {
            "key": cand.key, "mode": mode,
            "predicted": None if predicted is None else predicted.kind,
            "confidence": round(conf, 3), "model": self._model.stats(),
        }
        self._instr.log(action.name, cand.x, cand.y, mode, predicted, None,
                        self._model.stats(), {"key": cand.key})

        # guarda estado p/ o próximo passo
        self._prev_scene = scene
        self._last_key = cand.key
        self._last_predicted = predicted
        self._last_level = latest_frame.levels_completed or 0
        return action
