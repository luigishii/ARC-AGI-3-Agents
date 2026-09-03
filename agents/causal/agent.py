import os
import math
from collections import deque
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from .causal_model import CausalModel
from .instrumentation import Instrumentation
from .perception import match_objects, parse, to_grid, win_grid
from .policy import Policy, candidates
from .hud import HudMask
from .novelty import NoveltyModel, state_signature
from .transfer import shared_prior, abstract_feature, load_shared_once, DEFAULT_PRIOR_PATH
from .planning import TransitionModel, plan
from .navigate import MovementModel, navigate
from .perception_strategy import PerceptionStrategy
from .llm import (shared_llm_client, build_prompt, build_direct_prompt,
                  parse_goal, execute_goal, client_kind)
from .ranker import rank_candidates
from .ontology import LocalEffectTable, effect_signature
from .typed_model import TypedWorldModel, accept_rule
from .iw import iw_plan
from .goals import (compile_reward, static_reward_check, goal_fn_from_reward,
                    value_fn_from_reward, accept_reward, grounded_reward_fn,
                    grounded_multi_reward_fn, grounded_pattern_reward_fn)

QUERY_COOLDOWN = 15
GOAL_FAIL_MAX = 3
GOAL_AGE_MAX = 20
TYPE_MIN_OBS = 3          # transições mínimas p/ tentar sintetizar f_τ
TYPE_BUF_MAX = 64         # transições guardadas por tipo
TYPE_COOLDOWN = 8         # cadência esparsa da síntese de regra de tipo
REWARD_DEFER_MAX = 4      # ticks elegíveis a esperar o avatar antes de sintetizar a reward


def _obj_state(o) -> dict:
    """Estado mecânico serializável de um objeto p/ as regras f_τ (x=col, y=row)."""
    return {"x": int(round(o.centroid[1])), "y": int(round(o.centroid[0])),
            "color": int(o.color), "shape": o.shape_hash, "size": int(o.size)}


def _spatial_context(objects) -> str:
    """Bloco textual com posição (x,y), distâncias par-a-par (Manhattan) e resumo de
    grid 3x3 dos objetos — pro prompt de síntese de reward enxergar estrutura espacial
    (não só cor). x=col, y=row (mesma convenção de _obj_state)."""
    objs = list(objects)[:8]
    pts = [(int(round(o.centroid[1])), int(round(o.centroid[0]))) for o in objs]
    lines = ["OBJETOS (x=col, y=row):"]
    for i, o in enumerate(objs):
        lines.append(f"  id={i} color={int(o.color)} x={pts[i][0]} y={pts[i][1]} size={o.size}")
    pairs = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
            pairs.append((d, i, j))
    if pairs:
        lines.append("DISTANCIAS (Manhattan):")
        for d, i, j in pairs[:10]:
            lines.append(f"  d(id{i},id{j})={d}")
    G = 3
    cell = {}
    for i, o in enumerate(objs):
        gx = min(pts[i][0] * G // 64, G - 1)
        gy = min(pts[i][1] * G // 64, G - 1)
        cell.setdefault((gy, gx), int(o.color))
    lines.append("GRID 3x3 (cor do objeto por celula, '.'=vazio):")
    for gy in range(G):
        lines.append("  " + " ".join(str(cell.get((gy, gx), ".")) for gx in range(G)))
    return "\n".join(lines)


def _pick_target(objects, avatar_idx):
    """Alvo provável: objeto não-avatar, cor rara, compacto (não fundo/barra). Índice ou None."""
    objs = list(objects)[:8]
    cand = [i for i in range(len(objs)) if i != avatar_idx]
    if not cand:
        return None
    color_count = {}
    for o in objs:
        color_count[o.color] = color_count.get(o.color, 0) + 1
    largest = max(cand, key=lambda i: objs[i].size)
    cand2 = [i for i in cand if i != largest] or cand

    def _elongated(i):
        r0, c0, r1, c1 = objs[i].bbox
        w, h = c1 - c0 + 1, r1 - r0 + 1
        return max(w, h) / max(1, min(w, h)) >= 4

    cand3 = [i for i in cand2 if not _elongated(i)] or cand2
    av = objs[avatar_idx]
    ax, ay = av.centroid[1], av.centroid[0]

    def _key(i):
        o = objs[i]
        dist = abs(o.centroid[1] - ax) + abs(o.centroid[0] - ay)
        return (color_count[o.color], o.size, dist)

    return min(cand3, key=_key)


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
        self._llm = shared_llm_client(os.environ.get("QWEN_MODEL_PATH"))
        self._llm_on = os.environ.get("CAUSAL_LLM", "0") != "0"
        self._n_samples = int(os.environ.get("CAUSAL_SAMPLES", "1"))
        self._llm_kind = client_kind(self._llm)
        self._llm_calls = 0
        self._llm_max = int(os.environ.get("CAUSAL_LLM_MAX_CALLS", "100000"))
        self._repair_max = int(os.environ.get("CAUSAL_REPAIR", "1"))
        # Score-max Lever #2: politica de raciocinio direto passo-a-passo
        self._direct_on = os.environ.get("CAUSAL_DIRECT", "0") != "0"
        self._direct_cooldown = int(os.environ.get("CAUSAL_DIRECT_COOLDOWN", "2"))
        self._since_direct = 10 ** 9      # como _since_query: consulta no 1o frame elegivel
        self._last_effect_kind = None     # efeito real da ultima acao (feedback do prompt)
        self._direct_calls = 0
        self._direct_hits = 0
        if self._llm_on:      # log de boot: 'null' = LLM não subiu (degradou p/ fallback)
            print(f"[causal] LLM ativo: {self._llm_kind} (max_calls={self._llm_max})")
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
        self._buffer = deque(maxlen=int(os.environ.get("CAUSAL_BUFFER", "128")))
        self._last_key = None
        self._last_predicted = None
        self._last_level = 0
        self._seen_effects = set()
        self._pending_log = None
        self._hud = HudMask()
        self._prev_grid = None
        self._percept = PerceptionStrategy()
        # Fase-2: exploração por erro de ontologia (η) + world-model fatorado por tipo (f_τ)
        self._etable = LocalEffectTable()
        self._type_buffer = {}          # τ -> [transição {before,action,context,after}]
        self._typed = TypedWorldModel()
        self._eta_on = os.environ.get("CAUSAL_ETA", "0") != "0"
        self._typed_on = os.environ.get("CAUSAL_TYPED", "0") != "0"
        self._iw_on = os.environ.get("CAUSAL_IW", "0") != "0"
        self._since_type = 0
        self._reward_fn = None        # A: reward_function/predicado de meta (goal-directed IW)
        self._reward_src = None
        self._reward_rejected = 0     # diag: rewards barradas pelo check comportamental
        self._reward_defer = 0        # ticks já esperados pelo avatar (gate de síntese)
        self._iw_goal_calls = 0       # diag: IW rodou goal-directed (reward viva)
        self._iw_goal_hits = 0        # diag: IW achou caminho até a meta
        self._reward_real_true = 0    # diag: reward_fn deu goal_flag=True em cena real
        self._reward_real_evals = 0   # diag: cenas reais avaliadas pela reward_fn
        self._rprog = {}              # action_key -> deque[float] (deltas recentes de reward)
        self._rprog_fires = 0         # diag: vezes que a camada rprog escolheu a ação
        self._rprog_on = os.environ.get("CAUSAL_RPROG", "0") != "0"
        self._rprog_max = int(os.environ.get("CAUSAL_RPROG_MAX", "20"))
        self._cover = {}              # action_key -> nº de vezes tomada (exploração por cobertura)
        self._cover_on = os.environ.get("CAUSAL_COVER", "0") != "0"
        self._clickmap = os.environ.get("CAUSAL_CLICKMAP", "0") != "0"
        self._grounded = os.environ.get("CAUSAL_GROUNDED", "0") != "0"
        self._fix_run = 0             # repetições consecutivas da MESMA key escolhida
        self._fix_breaks = 0          # diag: vezes que o guarda quebrou uma fixação
        self._fix_on = os.environ.get("CAUSAL_FIX", "0") != "0"
        self._fix_k = int(os.environ.get("CAUSAL_FIX_K", "3"))
        # Transfer entre níveis: cores de objeto que foram produtivas (level_up)
        self._productive_colors = set()    # cores que estavam na key que causou level_up
        # Deteccao de "preso": acoes consecutivas sem mudanca de estado
        self._stale_count = 0
        self._stale_max = int(os.environ.get("CAUSAL_STALE_MAX", "30"))
        # Combo memory: sequencias de 2-3 acoes que produziram efeito positivo
        self._combo_buf = deque(maxlen=5)    # ultimas 5 keys
        self._combos = {}                     # (key1,key2) -> [delta_sum, count]
        self._combo_on = os.environ.get("CAUSAL_COMBO", "1") != "0"
        # LLM defer: roda N acoes heuristicas antes de permitir chamadas LLM.
        # Garante que TODOS os jogos iniciam rapidos (~20fps) e so usam LLM quando
        # as heuristicas (navigate, cover, grounded) ja tiveram chance de agir.
        self._llm_defer = int(os.environ.get("CAUSAL_LLM_DEFER", "50"))

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def cleanup(self, scorecard=None):
        if os.environ.get("CAUSAL_PRIOR_SAVE"):
            from .transfer import save_prior
            save_prior(self._prior, os.environ.get("CAUSAL_PRIOR", DEFAULT_PRIOR_PATH))
        stats = self.phase2_stats()
        print(f"[causal] phase2 stats: {stats}")   # B: diagnóstico no log
        try:                                        # e no OUTPUT (logs do rerun são ocultos)
            import json as _json
            with open("/kaggle/working/causal_phase2.json", "w") as _f:
                _json.dump(stats, _f)
        except Exception:
            pass
        super().cleanup(scorecard)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # full_reset sinaliza que o ambiente ja se resetou -> limpa a memoria interna,
        # mas NAO re-envia RESET (offline o jogo devolve full_reset=True em toda resposta
        # de RESET; re-resetar geraria loop infinito). RESET de verdade so quando o jogo
        # nao esta jogavel (NOT_PLAYED / GAME_OVER).
        need_reset = latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER)
        if need_reset or getattr(latest_frame, "full_reset", False):
            self._prev_scene = None
            self._last_key = None
            self._pending_log = None
            self._hud = HudMask()
            self._prev_grid = None
            self._last_effect_kind = None
            self._stale_count = 0
            # Limpa dados espaciais stale (layout muda entre tentativas).
            # productive_colors persiste (cross-nivel).
            self._rprog.clear()
            self._rprog_fires = 0
            self._cover.clear()
            self._combo_buf.clear()
            self._combos.clear()
            if need_reset:
                return GameAction.RESET

        grid = to_grid(latest_frame.frame)
        if self._prev_grid is not None:
            self._hud.update(self._prev_grid, grid)
        scene = match_objects(self._prev_scene, parse(latest_frame.frame, hud_mask=self._hud.mask()))
        self._percept.observe(len(scene.objects))   # §2: instabilidade → fallback grid (Tycho)
        self._eval_reward_real(scene)               # diag: reward_fn na cena real

        # fecha o loop causal da ação anterior
        if self._prev_scene is not None and self._last_key is not None:
            level_up = (latest_frame.levels_completed or 0) > self._last_level
            actual = self._model.observe(self._prev_scene, self._last_key, scene, level_up)
            self._model.record_prediction(self._last_predicted, actual)
            self._seen_effects.add(actual.kind)
            if self._last_feature is not None:
                self._prior.observe(self._last_feature, actual.kind)
            if level_up:
                # Transfer: registra a COR do objeto da key que resolveu o nivel.
                # No proximo nivel, objetos dessa cor ganham bonus no scoring.
                if self._last_key and "@c" in self._last_key:
                    try:
                        c = int(self._last_key.split("@c")[1].split("s")[0])
                        self._productive_colors.add(c)
                    except (ValueError, IndexError):
                        pass
                self._stale_count = 0
                # Limpa dados espaciais do nivel anterior (layout muda entre niveis).
                # productive_colors, _model e _move persistem (transferem entre niveis).
                self._rprog.clear()
                self._rprog_fires = 0
                self._cover.clear()
                self._combo_buf.clear()
                self._combos.clear()
                self._reward_fn = None       # re-sintetiza grounded pro novo layout
                self._reward_src = None
                self._reward_defer = 0
                win_scene = parse(win_grid(latest_frame.frame), hud_mask=self._hud.mask())
                self._novelty.record_goal_anchor(state_signature(win_scene))
                self._goal = None                     # nível cumprido → re-planejar
                self._last_effect_kind = None
            else:
                # transição DECISÃO→DECISÃO: alimenta retrodição + f_τ/η + movimento + forward
                self._buffer.append((self._prev_scene, self._last_key, actual.kind))
                self._last_effect_kind = actual.kind
                self._observe_types(self._prev_scene, scene, self._last_key)
                self._track_rprog(scene)          # B′: delta de reward real por ação
                self._track_cover()               # cobertura: conta a ação tomada
                # Deteccao de "preso": acoes sem efeito
                if actual.kind == "none":
                    self._stale_count += 1
                else:
                    self._stale_count = 0
                self._novelty.observe_transition(self._last_key, scene)
                self._move.observe(self._last_key, self._prev_scene, scene)
                cur_sig = state_signature(scene)
                if self._last_sig is not None:
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
        cands = candidates(scene, avail, clickmap=self._clickmap)
        keymap = {c.key: c for c in cands}
        moves = self._move.moves()
        # (1) consulta esparsa ao LLM: só se ligado, sem meta ativa e passado o cooldown.
        # Click-only: pula — LLM nao sabe raciocinar sobre coordenadas de pixel.
        self._since_query += 1
        self._since_direct += 1
        has_keyboard = any(not (a if isinstance(a, GameAction)
                                else GameAction.from_id(a)).is_complex() for a in avail)
        if (self._llm_on and has_keyboard and self._goal is None
                and self._since_query >= QUERY_COOLDOWN
                and self._llm_calls < self._llm_max and not self._direct_on
                and self.action_counter >= self._llm_defer):
            self._llm_calls += 1
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
                                       self._tmodel, self._novelty, moves,
                                       buffer=self._buffer, available=avail)
                self._goal = ({"type": "code", "source": best} if best
                              else (goals[0] if goals else None))
            else:
                self._goal = goals[0] if goals else None
            self._goal_age = 0
            self._since_query = 0
        # (1b) síntese esparsa de regra de tipo f_τ via LLM, validada por accept_rule.
        # Click-only: pula sintese LLM (mas _try_learn_reward ainda tenta grounded).
        self._since_type += 1
        if (self._typed_on and self._llm_on and has_keyboard
                and self._since_type >= TYPE_COOLDOWN
                and self._llm_calls < self._llm_max
                and self.action_counter >= self._llm_defer):
            tau = self._pick_type_to_learn()
            if tau is not None:
                self._try_learn_type_rule(tau)
            if self._should_learn_reward():
                self._try_learn_reward(scene)      # A: sintetiza predicado de meta
            self._since_type = 0
        # Grounded reward pra TODOS os jogos: sintetiza SEM LLM (calculada por heuristica).
        # Jogos de teclado tambem ganham reward grounded (manhattan, multi-align, pattern).
        if self._grounded and self._reward_fn is None:
            self._try_learn_reward(scene)
        cand = None
        # (-1) Sweep de tipos: garante que cada TIPO de ação (ACTION1..7) é testado
        # pelo menos 1x nos primeiros passos. Custa no máximo len(avail) ações mas
        # descobre todas as mecânicas (teclado, validar, undo). Sem isso o agente
        # colapsa em ACTION6 e nunca descobre que ACTION5=validar ou ACTION7=undo.
        if self.action_counter < len(avail) * 2:
            for a in avail:
                act = a if isinstance(a, GameAction) else GameAction.from_id(a)
                k = act.name
                if k not in self._cover:
                    cand = keymap.get(k)
                    if cand is not None:
                        break
        # (2b) two-phase: se a ultima acao foi clique com efeito (selecao/toggle),
        # tenta a proxima fase — teclado (ka59, ar25, cn04, sp80) OU outro clique
        # num objeto diferente (r11l: click peca → click destino; lf52: peca→marker).
        if (cand is None and self._last_key is not None
                and "@c" in (self._last_key or "")
                and self._last_effect_kind not in (None, "none")):
            if has_keyboard:
                # Fase 2 = teclado: tenta cada direcao
                for k in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
                    if k in keymap and self._cover.get(k, 0) < 3:
                        cand = keymap[k]
                        break
            if cand is None:
                # Fase 2 = outro clique: escolhe objeto de cor/posicao diferente
                alt = [c for c in cands if c.key != self._last_key
                       and c.has_object and c.key != "ACTION6@bg"]
                if alt:
                    best = min(alt, key=lambda c: self._cover.get(c.key, 0))
                    cand = best
        # (3) fallback determinístico: navigate → plan → greedy
        if cand is None and self._nav_on:
            nk = navigate(scene, self._move)
            if nk is not None:
                cand = keymap.get(nk)
        if cand is None and self._rprog_on and cands:
            rk = self._rprog_decide(cands)    # B′: progresso model-free por reward real
            if rk is not None:
                cand = keymap.get(rk)
        # Combo: se a ultima key faz parte de um combo positivo, executa a 2a key
        if cand is None and self._combo_on and self._last_key:
            best_combo_key, best_avg = None, 0.0
            for (k1, k2), cr in self._combos.items():
                if k1 == self._last_key and cr[1] >= 2 and cr[0] / cr[1] > best_avg:
                    if k2 in keymap:
                        best_avg = cr[0] / cr[1]
                        best_combo_key = k2
            if best_combo_key is not None:
                cand = keymap[best_combo_key]
        # ACTION7 reativo: undo quando a ultima acao piorou o reward (delta < 0).
        # Cobre ar25, lf52, sb26, sk48, su15 — jogos com undo estrategico.
        if (cand is None and self._last_key is not None
                and "ACTION7" in keymap and self._last_effect_kind not in (None, "none")):
            row = self._rprog.get(self._last_key)
            if row and len(row) >= 1 and row[-1] < -0.5:
                cand = keymap["ACTION7"]
        # ACTION5 periodico: validar/grab/ciclar/disparar. Muitos jogos exigem
        # ACTION5 apos montar o estado (sb26=run, wa30=grab, cd82=fire, sp80=flow).
        # Tenta a cada 15 acoes se houve mudanca de estado (efeito != none recente).
        if (cand is None and "ACTION5" in keymap
                and self.action_counter >= 8
                and self.action_counter % 15 < 2
                and any(e != "none" for e in list(self._seen_effects)[:5])):
            cand = keymap["ACTION5"]
        if cand is None and self._iw_on and cands:
            ik = self._iw_decide(scene, cands)     # IW sobre o TypedWorldModel (f_τ)
            if ik is not None:
                cand = keymap.get(ik)
        # --- LLM strategies (so apos _llm_defer acoes heuristicas) ---
        if cand is None and self._direct_on:
            cand = self._direct_decide(scene, avail, keymap, moves)
        if cand is None and self._goal is not None:
            self._goal_age += 1
            gkey = execute_goal(self._goal, scene, moves)
            if gkey is not None and gkey in keymap:
                cand = keymap[gkey]
                self._goal_fails = 0
                if self._goal.get("type") == "press":
                    self._goal = None
            else:
                self._goal_fails += 1
            if self._goal is not None and (
                self._goal_fails >= GOAL_FAIL_MAX or self._goal_age >= GOAL_AGE_MAX
            ):
                self._goal = None
        if cand is None and self._plan_on and cands:
            planned = plan(state_signature(scene), [c.key for c in cands],
                           self._tmodel, self._novelty, self._novelty.goal_anchors)
            if planned is not None:
                cand = keymap.get(planned)
        if cand is None and self._eta_on and cands:
            ek = self._eta_explore(cands)     # sonda a ação de linha mais ambígua (η alto)
            if ek is not None:
                cand = keymap.get(ek)
        if cand is None and self._cover_on and cands:
            ck = self._cover_decide(cands)    # exploração por cobertura (anti-fixação)
            if ck is not None:
                cand = keymap.get(ck)
        if cand is None:
            cand = self._policy.decide(
                scene, self._model, avail,
                self._seen_effects, budget_frac, novelty=self._novelty, prior=self._prior,
                clickmap=self._clickmap,
                rprog=self._rprog if self._rprog_on else None,
                productive_colors=self._productive_colors or None,
            )
        if cand is None:
            self._pending_log = None
            return GameAction.RESET
        # Reset voluntário: se preso (N ações sem efeito), reseta o nível.
        # Stale_max adaptativo: proporcional ao numero de candidatos (mais opcoes
        # = mais paciencia). Min 15, max _stale_max (default 30).
        adaptive_stale = max(15, min(self._stale_max, len(cands) * 3))
        if self._stale_count >= adaptive_stale:
            self._stale_count = 0
            self._pending_log = None
            return GameAction.RESET
        cand = self._antifix(cand, cands, keymap)   # guarda global anti-fixação
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
            "percept_mode": self._percept.mode(),
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

    # --- Fase-2: exploração por η + world-model fatorado por tipo (f_τ) ---
    def _observe_types(self, prev_scene, scene, key) -> None:
        """Arquiva, por objeto pareado (id), a assinatura de efeito (η) e a transição
        mecânica (before→after) sob o tipo τ = shape_hash do objeto."""
        prev_by_id = {o.id: o for o in prev_scene.objects}
        for o in scene.objects:
            po = prev_by_id.get(o.id)
            if po is None:
                continue
            tau = po.shape_hash
            self._etable.observe(tau, key, "", effect_signature(po, o))
            buf = self._type_buffer.setdefault(tau, [])
            buf.append({"before": _obj_state(po), "action": key,
                        "context": {}, "after": _obj_state(o)})
            if len(buf) > TYPE_BUF_MAX:
                buf.pop(0)

    def _iw_decide(self, scene, cands):
        """Planeja com Iterated Width sobre o TypedWorldModel (regras f_τ aceitas).
        Sem regras aceitas → None (cai no fallback). Com reward aprendida → best-first
        que sobe o reward denso (value_fn); sem reward → exploração width-based."""
        if not self._typed.sources:
            return None
        start = [(o.shape_hash, _obj_state(o)) for o in scene.objects]
        vf = value_fn_from_reward(self._reward_fn) if self._reward_fn else None
        r = iw_plan(start, [c.key for c in cands], self._typed, value_fn=vf, max_nodes=300)
        if vf is not None:                       # diag: IW value-directed disparou
            self._iw_goal_calls += 1
            if r is not None:                    # achou ação que melhora o valor
                self._iw_goal_hits += 1
        return r

    def _direct_decide(self, scene, avail, keymap, moves):
        """Score-max Lever #2: consulta o LLM pela PROXIMA acao imediata (esparso via
        cooldown), valida contra keymap (available-guard) e devolve o candidato. Miss ou
        erro -> None (cai na pilha deterministica). Exception-safe."""
        if self._since_direct < self._direct_cooldown:
            return None
        if self._llm_calls >= self._llm_max:
            return None
        if self.action_counter < self._llm_defer:
            return None
        # Click-only: LLM nao sabe raciocinar sobre "clique no pixel X,Y"
        # — so gasta ~60s por chamada com 0 hits. Pula e deixa o policy decidir.
        has_keyboard = any(not (a if isinstance(a, GameAction)
                                else GameAction.from_id(a)).is_complex() for a in avail)
        if not has_keyboard:
            return None
        self._since_direct = 0
        self._llm_calls += 1
        self._direct_calls += 1
        dyn = {"available": [str(a) for a in avail], "moves": moves, "notes": ""}
        last = {"key": self._last_key, "effect": self._last_effect_kind}
        try:
            resp = self._llm.complete(build_direct_prompt(scene, dyn, last))
            g = parse_goal(resp)
            key = execute_goal(g, scene, moves) if g is not None else None
        except Exception:
            return None
        cand = keymap.get(key) if key is not None else None
        if cand is not None:
            self._direct_hits += 1
        return cand

    def _eval_reward_real(self, scene):
        """Diag (b)vs(c): avalia a reward_fn aprendida na cena REAL observada.
        Reusa goal_fn_from_reward (à prova de exceção). Não altera decisão."""
        if self._reward_fn is None:
            return
        state = [(o.shape_hash, _obj_state(o)) for o in scene.objects]
        self._reward_real_evals += 1
        if goal_fn_from_reward(self._reward_fn)(state):
            self._reward_real_true += 1

    def _track_rprog(self, scene):
        """B′: acumula Δ do reward REAL (depois-antes) por action_key numa janela
        deslizante (deque maxlen=10). Evita fixação por sinal antigo positivo que
        já saturou. Model-free. Só decisão→decisão."""
        if self._reward_fn is None or self._last_key is None or self._prev_scene is None:
            return
        vf = value_fn_from_reward(self._reward_fn)
        before = [(o.shape_hash, _obj_state(o)) for o in self._prev_scene.objects]
        after = [(o.shape_hash, _obj_state(o)) for o in scene.objects]
        vb, va = vf(before), vf(after)
        if not (math.isfinite(vb) and math.isfinite(va)):
            return
        row = self._rprog.setdefault(self._last_key, deque(maxlen=10))
        delta = va - vb
        row.append(delta)
        # Combo memory: registra delta da SEQUENCIA (key_anterior, key_atual)
        if self._combo_on and len(self._combo_buf) >= 1:
            pair = (self._combo_buf[-1], self._last_key)
            cr = self._combos.setdefault(pair, [0.0, 0])
            cr[0] += delta
            cr[1] += 1
        self._combo_buf.append(self._last_key)

    def _rprog_decide(self, cands):
        """B′: escolhe a ação de maior Δ médio de reward real (>0). Min 3 observações
        por key, cap de _rprog_max fires por episódio (evita monopolizar). Sem
        dados/positivo/cap estourado → None."""
        if self._rprog_fires >= self._rprog_max:
            return None
        best_key, best_avg = None, 0.0
        for c in cands:
            row = self._rprog.get(c.key)
            if not row or len(row) < 3:
                continue
            avg = sum(row) / len(row)
            if avg > best_avg:
                best_avg, best_key = avg, c.key
        if best_key is not None:
            self._rprog_fires += 1
        return best_key

    def _track_cover(self):
        """Conta a ação efetivamente tomada (p/ o sweep de cobertura). Só decisão→decisão."""
        if self._last_key is not None:
            self._cover[self._last_key] = self._cover.get(self._last_key, 0) + 1

    def _cover_decide(self, cands):
        """Exploração por cobertura: escolhe a candidata MENOS visitada. Prioriza
        TIPOS de ação não-testados (keyboard/ACTION5/7 > click repetido),
        has_object antes de vazio, objetos pequenos, evita repetir a última ação."""
        def rank(c):
            visits = self._cover.get(c.key, 0)
            # Tipos de ação não-testados (ACTION1-5,7) tem prioridade sobre
            # variantes de clique já vistas — garante que teclado/validar/undo
            # são descobertos cedo em jogos mistos.
            is_keyboard = c.x is None   # keyboard actions tem x=None
            type_prio = 0 if (is_keyboard and visits == 0) else 1
            return (visits, type_prio,
                    0 if c.has_object else 1,
                    1 if c.key == self._last_key else 0,
                    c.obj_size)
        return min(cands, key=rank).key

    def _antifix(self, cand, cands, keymap):
        """Guarda GLOBAL anti-fixação: se a mesma key repete >= FIX_K vezes, sobrepõe a
        decisão por uma escolha de cobertura (menos-visitada) EXCLUINDO a key fixada.
        Pega fixação venha de qual camada da pilha for."""
        if cand.key == self._last_key:
            self._fix_run += 1
        else:
            self._fix_run = 0
        if self._fix_on and self._fix_run >= self._fix_k and cands:
            alt = [c for c in cands if c.key != cand.key]
            if alt:
                cand = keymap.get(self._cover_decide(alt), cand)
                self._fix_breaks += 1
                self._fix_run = 0
        return cand

    def _eta_bonus(self, key) -> float:
        # η = ontology_error(0, effect_entropy) = incerteza de efeito da linha (τ,key).
        # (type_entropy=0 até existir um classificador de tipos — só then η fica completo.)
        best = 0.0
        for j in self._etable.rows:
            if j[1] == key:
                best = max(best, self._etable.row_entropy(j))
        return best

    def _eta_explore(self, cands):
        best_key, best_eta = None, 0.0
        for c in cands:
            e = self._eta_bonus(c.key)
            if e > best_eta:
                best_eta, best_key = e, c.key
        return best_key

    def _pick_type_to_learn(self):
        """Tipo mais DETERMINÍSTICO (menor η) com dados suficientes e sem regra aceita —
        é o que o Qwen consegue modelar; η alto fica p/ sondagem/exploração."""
        best, best_eta = None, None
        for tau, buf in self._type_buffer.items():
            if tau in self._typed.sources or len(buf) < TYPE_MIN_OBS:
                continue
            eta = max((self._etable.row_entropy(j) for j in self._etable.rows
                       if j[0] == tau), default=0.0)
            if best_eta is None or eta < best_eta:
                best_eta, best = eta, tau
        return best

    def _build_type_prompt(self, tau, transitions) -> str:
        lines = [f"Tipo {tau}: infira transition(obj, action, ctx) que reproduz exatamente:"]
        for t in transitions[:8]:
            lines.append(f"  {t['before']} --{t['action']}--> {t['after']}")
        lines.append('Responda JSON {"type":"code","source":"def transition(obj, action, ctx): ..."}')
        return "\n".join(lines)

    def _build_reward_prompt(self, scene) -> str:
        ctx = _spatial_context(scene.objects)
        aid = self._move.avatar_id()
        avatar_idx = None
        if aid is not None:
            for i, o in enumerate(list(scene.objects)[:8]):
                if o.id == aid:
                    avatar_idx = i
                    break
        target_idx = (_pick_target(scene.objects, avatar_idx)
                      if avatar_idx is not None else None)
        ai = avatar_idx if avatar_idx is not None else 0
        if avatar_idx is not None and target_idx is not None:
            ground = (f"OBJETO CONTROLAVEL (avatar) = state[{avatar_idx}]. "
                      f"ALVO PROVAVEL = state[{target_idx}] (objeto raro/compacto, distinto do avatar). "
                      f"A reward DEVE medir a distancia state[{avatar_idx}] -> state[{target_idx}].\n")
            few = (
                f"  # avatar (state[{avatar_idx}]) se aproxima do alvo (state[{target_idx}])\n"
                "  def reward_function(state):\n"
                "      pts=[b for _,b in state]\n"
                f"      if len(pts)<={max(avatar_idx, target_idx)}: return (0.0, False)\n"
                f"      a=pts[{avatar_idx}]; t=pts[{target_idx}]\n"
                "      d=abs(a['x']-t['x'])+abs(a['y']-t['y'])\n"
                "      return (-float(d), d==0)\n"
            )
        elif avatar_idx is not None:
            ground = (f"OBJETO CONTROLAVEL (avatar) = state[{avatar_idx}]; a reward DEVE "
                      f"medir a distancia DELE (state[{avatar_idx}]) ate o alvo.\n")
            few = (
                f"  # distancia: o avatar (state[{ai}]) se aproxima do alvo mais proximo\n"
                "  def reward_function(state):\n"
                "      pts=[b for _,b in state]\n"
                "      if len(pts)<2: return (0.0, False)\n"
                f"      a=pts[{ai}]\n"
                f"      others=[c for k,c in enumerate(pts) if k!={ai}]\n"
                "      d=min(abs(a['x']-c['x'])+abs(a['y']-c['y']) for c in others)\n"
                "      return (-float(d), d==0)\n"
            )
        else:
            ground = ""
            few = (
                "  # distancia: objeto[0] se aproxima do alvo mais proximo\n"
                "  def reward_function(state):\n"
                "      pts=[b for _,b in state]\n"
                "      if len(pts)<2: return (0.0, False)\n"
                "      a=pts[0]\n"
                "      d=min(abs(a['x']-b['x'])+abs(a['y']-b['y']) for b in pts[1:])\n"
                "      return (-float(d), d==0)\n"
            )
        return (
            "Infira reward_function(state) que retorna (reward, goal_flag). REGRAS: "
            "(1) reward é um número GRADUADO — maior = mais perto de resolver, NÃO use só 0/1; "
            "(2) NÃO hardcode tamanhos/posições exatos (magic numbers) — use relações/distâncias; "
            "(3) goal_flag=True SÓ quando o nível está realmente resolvido (raro). "
            "A META costuma ser ESPACIAL: aproximar/alinhar um objeto de um alvo, ou casar um "
            "arranjo de células — use POSIÇÃO (x,y) e DISTÂNCIAS, não contagem de cor. "
            "O state é lista de (tipo,{x,y,color,shape}); x=col, y=row.\n"
            f"{ctx}\n"
            f"{ground}"
            "EXEMPLOS (reward espacial, só usam o state, sem import):\n"
            f"{few}"
            "  # arranjo: quantos objetos alinhados na mesma linha (y) do alvo\n"
            "  def reward_function(state):\n"
            "      pts=[b for _,b in state]\n"
            "      if not pts: return (0.0, False)\n"
            "      ys=[b['y'] for b in pts]\n"
            "      return (float(sum(1 for y in ys if y==ys[0])), False)\n"
            'Responda SÓ JSON {"type":"code","source":"def reward_function(state): ..."}'
        )

    def _should_learn_reward(self) -> bool:
        """Gate: só sintetiza a reward quando o avatar já foi aprendido (grounding entra no
        prompt) OU após REWARD_DEFER_MAX ticks (deadline p/ jogos sem avatar/clique)."""
        if self._reward_fn is not None:
            return False
        if self._move.avatar_id() is not None or self._reward_defer >= REWARD_DEFER_MAX:
            return True
        self._reward_defer += 1
        return False

    def _try_learn_reward(self, scene) -> bool:
        """A: sintetiza a reward via LLM e ACEITA a 1ª que passa o check estático
        (anti-trapaça) E o comportamental (accept_reward: sem constante/sempre-True em
        estados reais). Self-repair com o motivo da rejeição até CAUSAL_REPAIR vezes."""
        if self._reward_fn is not None:
            return False
        # GROUNDED: reward CALCULADA (nao chutada pelo LLM). Se o avatar (aprendido) e
        # um alvo (raro/compacto) sao conhecidos, usa -manhattan(avatar,alvo) por cor.
        # A vitoria no AGI-3 e sempre espacial (docs/GAMES.md), nunca contagem-de-cor.
        if self._grounded:
            aid = self._move.avatar_id()
            objs = list(scene.objects)
            avatar_idx = next((i for i, o in enumerate(objs[:8]) if o.id == aid), None)
            if avatar_idx is not None:
                tidx = _pick_target(scene.objects, avatar_idx)
                if tidx is not None:
                    ac, tc = objs[avatar_idx].color, objs[tidx].color
                    self._reward_fn = grounded_reward_fn(ac, tc)
                    self._reward_src = f"grounded:-dist(cor{ac}->cor{tc})"
                    return True
            # sem avatar claro (classe ALINHAMENTO: sokoban/manipulacao) -> reward
            # multi-objeto por cor (marcador<->alvo), excluindo HUD/parede grandes.
            same = sum(1 for c in set(o.color for o in objs)
                       if sum(1 for o in objs if o.color == c and o.size <= 64) >= 2)
            if same:
                self._reward_fn = grounded_multi_reward_fn()
                self._reward_src = "grounded:multi-align(same-color,small)"
                return True
            # classe PINTURA/SEQUENCIA/FLUXO: referencia (topo) vs editavel (baixo).
            tops = sum(1 for o in objs if o.size <= 64 and o.centroid[0] < 32)
            bots = sum(1 for o in objs if o.size <= 64 and o.centroid[0] >= 32)
            if tops and bots:
                self._reward_fn = grounded_pattern_reward_fn()
                self._reward_src = "grounded:pattern(topo-vs-baixo)"
                return True
            # grounded on mas sem estrutura util -> so cai no LLM apos o deferimento
            if self.action_counter < self._llm_defer or self._llm_calls >= self._llm_max:
                return False
        states = [[(o.shape_hash, _obj_state(o)) for o in sc.objects]
                  for (sc, _k, _e) in self._buffer]
        states.append([(o.shape_hash, _obj_state(o)) for o in scene.objects])
        prompt = self._build_reward_prompt(scene)
        for _ in range(self._repair_max + 1):
            self._llm_calls += 1
            resps = (self._llm.complete_many(prompt, self._n_samples)
                     if self._n_samples > 1 else [self._llm.complete(prompt)])
            last_err = "sem resposta"
            for r in resps:
                g = parse_goal(r)
                src = g.get("source") if g and g.get("type") == "code" else None
                if not src or not static_reward_check(src):
                    last_err = "não passa no check estático (usa o state? sem global?)"
                    continue
                ok, reason = accept_reward(src, states)
                if ok:
                    self._reward_fn = compile_reward(src)
                    self._reward_src = src
                    return True
                self._reward_rejected += 1
                last_err = reason
            prompt = self._build_reward_repair_prompt(scene, last_err)
        return False

    def _build_reward_repair_prompt(self, scene, err) -> str:
        base = self._build_reward_prompt(scene)
        return base + f"\nA tentativa anterior FOI REJEITADA: {err}. Corrija e responda SÓ o JSON."

    def phase2_stats(self) -> dict:
        """B: telemetria diagnosticável no log do Kaggle."""
        return {
            "llm_kind": self._llm_kind,
            "llm_calls": self._llm_calls,
            "n_types": len(self._type_buffer),
            "n_rules": len(self._typed.sources),
            "reward_learned": self._reward_fn is not None,
            "reward_src": self._reward_src,
            "reward_rejected": self._reward_rejected,
            "cover_keys": len(self._cover),
            "fix_breaks": self._fix_breaks,
            "iw_goal_calls": self._iw_goal_calls,
            "iw_goal_hits": self._iw_goal_hits,
            "reward_real_true": self._reward_real_true,
            "reward_real_evals": self._reward_real_evals,
            "rprog_actions": sum(1 for r in self._rprog.values() if len(r) >= 3 and sum(r) / len(r) > 0),
            "rprog_fires": self._rprog_fires,
            "direct_calls": self._direct_calls,
            "direct_hits": self._direct_hits,
            "eta_rows": len(self._etable.rows),
            "productive_colors": sorted(self._productive_colors),
            "stale_count": self._stale_count,
        }

    def _rule_error(self, src) -> str:
        from .typed_model import compile_rule
        if compile_rule(src) is None:
            return "não compila (falta 'def transition(obj, action, ctx)' ou erro de sintaxe)"
        return "compila mas não reproduz exatamente as transições (replay mismatch)"

    def _build_repair_prompt(self, tau, transitions, err) -> str:
        base = self._build_type_prompt(tau, transitions)
        return base + f"\nA tentativa anterior FALHOU: {err}. Corrija e responda SÓ o JSON."

    def _try_learn_type_rule(self, tau) -> bool:
        """Consulta o LLM por f_τ e ACEITA a 1ª regra que passa o replay exato das
        transições daquele tipo (accept_rule). Self-repair barato: se nenhuma amostra
        passa, realimenta o erro e re-pergunta até CAUSAL_REPAIR vezes."""
        buf = self._type_buffer.get(tau, [])
        if tau in self._typed.sources or len(buf) < TYPE_MIN_OBS:
            return False
        prompt = self._build_type_prompt(tau, buf)
        for _ in range(self._repair_max + 1):
            self._llm_calls += 1
            if self._n_samples > 1:
                resps = self._llm.complete_many(prompt, self._n_samples)
            else:
                resps = [self._llm.complete(prompt)]
            last_err = "sem resposta"
            for r in resps:
                g = parse_goal(r)
                src = g.get("source") if g and g.get("type") == "code" else None
                if not src:
                    last_err = "resposta não é JSON de código válido"
                    continue
                if accept_rule(src, buf):
                    self._typed.set_rule(tau, src)
                    return True
                last_err = self._rule_error(src)
            prompt = self._build_repair_prompt(tau, buf, last_err)   # realimenta o erro
        return False
