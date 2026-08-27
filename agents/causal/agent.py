import os
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from .causal_model import CausalModel
from .instrumentation import Instrumentation
from .perception import match_objects, parse, to_grid
from .policy import Policy, candidates
from .hud import HudMask
from .novelty import NoveltyModel, state_signature
from .transfer import shared_prior, abstract_feature, load_shared_once, DEFAULT_PRIOR_PATH
from .planning import TransitionModel, plan
from .navigate import MovementModel, navigate
from .llm import make_llm_client, build_prompt, parse_goal, execute_goal
from .ranker import rank_candidates

QUERY_COOLDOWN = 8
GOAL_FAIL_MAX = 3
GOAL_AGE_MAX = 20


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
        self._tmodel = TransitionModel()
        self._move = MovementModel()
        self._nav_on = os.environ.get("CAUSAL_NAV", "1") != "0"
        self._llm = make_llm_client(os.environ.get("QWEN_MODEL_PATH"))
        self._llm_on = os.environ.get("CAUSAL_LLM", "0") != "0"
        self._n_samples = int(os.environ.get("CAUSAL_SAMPLES", "1"))
        self._goal = None
        self._goal_age = 0
        self._goal_fails = 0
        self._since_query = 10 ** 9
        self._last_sig = None
        self._plan_on = os.environ.get("CAUSAL_PLAN", "1") != "0"
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
                self._goal = None                     # nível cumprido → re-planejar
            self._novelty.observe_transition(self._last_key, scene)
            self._move.observe(self._last_key, self._prev_scene, scene)
            if self._last_feature is not None:
                self._prior.observe(self._last_feature, actual.kind)
            cur_sig = state_signature(scene)
            if self._last_sig is not None and self._last_key is not None:
                self._tmodel.observe(self._last_sig, self._last_key, cur_sig)
            # logging deferido: agora sabemos o efeito real da ação anterior
            if self._pending_log is not None:
                self._instr.log(**self._pending_log, actual=actual)
                self._pending_log = None

        # decide a próxima ação
        budget_frac = 1.0
        if self.MAX_ACTIONS not in (0, float("inf")):
            budget_frac = max(0.0, 1 - self.action_counter / self.MAX_ACTIONS)
        avail = latest_frame.available_actions or [GameAction.ACTION1]
        cands = candidates(scene, avail)
        keymap = {c.key: c for c in cands}
        moves = self._move.moves()
        # (1) consulta esparsa ao LLM: só se ligado, sem meta ativa e passado o cooldown
        self._since_query += 1
        if self._llm_on and self._goal is None and self._since_query >= QUERY_COOLDOWN:
            dyn = {"available": [str(a) for a in avail], "moves": moves, "notes": ""}
            prompt = build_prompt(scene, dyn)
            if self._n_samples > 1:
                resps = self._llm.complete_many(prompt, self._n_samples)
            else:
                resps = [self._llm.complete(prompt)]
            goals = [g for g in (parse_goal(r) for r in resps) if g is not None]
            code_srcs = [g["source"] for g in goals if g.get("type") == "code"]
            if code_srcs:
                best = rank_candidates(code_srcs, scene, self._model,
                                       self._tmodel, self._novelty, moves)
                self._goal = ({"type": "code", "source": best} if best
                              else (goals[0] if goals else None))
            else:
                self._goal = goals[0] if goals else None
            self._goal_age = 0
            self._since_query = 0
        cand = None
        # (2) meta do LLM com validação
        if self._goal is not None:
            self._goal_age += 1
            gkey = execute_goal(self._goal, scene, moves)
            if gkey is not None and gkey in keymap:
                cand = keymap[gkey]
                self._goal_fails = 0
            else:
                self._goal_fails += 1
            if self._goal_fails >= GOAL_FAIL_MAX or self._goal_age >= GOAL_AGE_MAX:
                self._goal = None
        # (3) fallback determinístico: navigate → plan → greedy
        if cand is None and self._nav_on:
            nk = navigate(scene, self._move)
            if nk is not None:
                cand = keymap.get(nk)
        if cand is None and self._plan_on and cands:
            planned = plan(state_signature(scene), [c.key for c in cands],
                           self._tmodel, self._novelty, self._novelty.goal_anchors)
            if planned is not None:
                cand = keymap.get(planned)
        if cand is None:
            cand = self._policy.decide(
                scene, self._model, avail,
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
        self._last_sig = state_signature(scene)
        self._last_feature = abstract_feature(cand)
        self._last_predicted = predicted
        self._last_level = latest_frame.levels_completed or 0
        return action
