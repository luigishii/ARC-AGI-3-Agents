import os
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from .causal_model import CausalModel
from .instrumentation import Instrumentation
from .perception import match_objects, parse, to_grid
from .policy import Policy
from .hud import HudMask
from .novelty import NoveltyModel, state_signature
from .transfer import shared_prior, abstract_feature, load_shared_once, DEFAULT_PRIOR_PATH


class CausalObjectAgent(Agent):
    """Agente objeto-cêntrico causal (v1). Ver docs/superpowers/specs/2026-08-27-causal-object-agent-design.md."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_causal_state()

    def _init_causal_state(self) -> None:
        self.MAX_ACTIONS = int(os.environ.get("CAUSAL_MAX_ACTIONS", type(self).MAX_ACTIONS))
        self._model = CausalModel()
        self._novelty = NoveltyModel()
        self._prior = shared_prior()
        load_shared_once(os.environ.get("CAUSAL_PRIOR", DEFAULT_PRIOR_PATH))
        self._last_feature = None
        self._policy = Policy(seed=0, epsilon=0.05)
        self._instr = Instrumentation(path=os.environ.get("CAUSAL_LOG"))
        self._prev_scene = None
        self._last_key = None
        self._last_predicted = None
        self._last_level = 0
        self._seen_effects = set()
        self._pending_log = None
        self._hud = HudMask()
        self._prev_grid = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def cleanup(self, scorecard=None):
        if os.environ.get("CAUSAL_PRIOR_SAVE"):
            from .transfer import save_prior
            save_prior(self._prior, os.environ.get("CAUSAL_PRIOR", DEFAULT_PRIOR_PATH))
        super().cleanup(scorecard)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER) or getattr(
            latest_frame, "full_reset", False
        ):
            self._prev_scene = None
            self._last_key = None
            self._pending_log = None
            self._hud = HudMask()
            self._prev_grid = None
            return GameAction.RESET

        grid = to_grid(latest_frame.frame)
        if self._prev_grid is not None:
            self._hud.update(self._prev_grid, grid)
        scene = match_objects(self._prev_scene, parse(latest_frame.frame, hud_mask=self._hud.mask()))

        # fecha o loop causal da ação anterior
        if self._prev_scene is not None and self._last_key is not None:
            level_up = (latest_frame.levels_completed or 0) > self._last_level
            actual = self._model.observe(self._prev_scene, self._last_key, scene, level_up)
            self._model.record_prediction(self._last_predicted, actual)
            self._seen_effects.add(actual.kind)
            if level_up:
                self._novelty.record_goal_anchor(state_signature(self._prev_scene))
            self._novelty.observe_transition(self._last_key, scene)
            if self._last_feature is not None:
                self._prior.observe(self._last_feature, actual.kind)
            # logging deferido: agora sabemos o efeito real da ação anterior
            if self._pending_log is not None:
                self._instr.log(**self._pending_log, actual=actual)
                self._pending_log = None

        # decide a próxima ação
        budget_frac = 1.0
        if self.MAX_ACTIONS not in (0, float("inf")):
            budget_frac = max(0.0, 1 - self.action_counter / self.MAX_ACTIONS)
        cand = self._policy.decide(
            scene, self._model, latest_frame.available_actions or [GameAction.ACTION1],
            self._seen_effects, budget_frac, novelty=self._novelty, prior=self._prior,
        )
        if cand is None:
            self._pending_log = None
            return GameAction.RESET
        action = cand.action
        if action.is_complex():
            action.set_data({"x": cand.x, "y": cand.y})

        predicted, conf = self._model.predict(cand.key)
        mode = "EXPLOIT" if self._model.is_progress(cand.key) else "EXPLORE"
        action.reasoning = {
            "key": cand.key, "mode": mode,
            "predicted": None if predicted is None else predicted.kind,
            "confidence": round(conf, 3), "model": self._model.stats(),
            "novelty_yield": round(self._novelty.yield_estimate(cand.key), 3),
        }
        # guarda o registro (sem `actual`); será logado quando o efeito for observado
        self._pending_log = {
            "action_name": action.name,
            "x": cand.x,
            "y": cand.y,
            "mode": mode,
            "predicted": predicted,
            "model_stats": self._model.stats(),
            "reasoning": {"key": cand.key},
        }

        # guarda estado p/ o próximo passo
        self._prev_scene = scene
        self._prev_grid = grid
        self._last_key = cand.key
        self._last_feature = abstract_feature(cand)
        self._last_predicted = predicted
        self._last_level = latest_frame.levels_completed or 0
        return action
