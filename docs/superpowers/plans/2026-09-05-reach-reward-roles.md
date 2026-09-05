# Reward de alcance com papéis aprendidos na vitória — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dar gradiente à reward do nível seguinte quando a reward validada pela vitória fica plana: aprender na vitória o par (móvel, pouso) e, no nível novo, medir a distância do móvel (aprendido pelo `MovementModel`) ao objeto da cor do pouso.

**Architecture:** Módulo puro `roles.py` (papéis da vitória) + uma reward grounded nova em `goals.py` + wiring em `agent.py` sob `CAUSAL_REACH` (detector de planura → adoção validada por `accept_reward_fn`). Default off é byte-idêntico; com o flag, nada muda no 1º nível de nenhum jogo.

**Tech Stack:** Python 3.12, numpy, pytest (`uv run pytest`), `arcengine` (GameAction/GameState nos testes de agente). Spec: `docs/superpowers/specs/2026-09-05-reach-reward-roles-design.md`.

## Global Constraints

- Toggle `CAUSAL_REACH` (env), default off; caminho off byte-idêntico.
- Papéis só existem após ≥1 level-up → 1º nível de qualquer jogo inalterado com o flag on.
- Constante `REACH_FLAT_K = 8` (passos com efeito visível sem mudança da reward).
- Exception-safe: `learn_win_roles` → `None`; reward → `(0.0, False)`.
- Nenhum conhecimento por `game_id`. Comentários em pt-BR, commits em inglês (prefixo feat/fix/docs/test).
- Rodar `uv run pytest tests/causal tests/kaggle -x -q` antes de cada commit; base = 519 verdes.
- Trabalhar em worktree/branch `reach-reward` a partir de `main` (`1ec26a2`).

---

## File Structure

- Create `agents/causal/roles.py` — `WinRoles`, `learn_win_roles` (puro, sem estado).
- Modify `agents/causal/goals.py` — `grounded_reach_reward_fn` (após `grounded_reward_fn`).
- Modify `agents/causal/agent.py` — estado, level-up (papéis), fecha-loop (planura), adoção, telemetria.
- Modify `kaggle/build_notebook.py` (`MODULES` + `ENV`), `kaggle/build_offline_notebook.py` (`OFFLINE_ENV`).
- Create `analysis/reach_probe.py` — probe de validação sobre recording.
- Tests: `tests/causal/test_roles.py`, `tests/causal/test_grounded_reach.py`, `tests/causal/test_agent_reach.py`, `tests/kaggle/test_build_notebook.py`, `tests/kaggle/test_build_offline_notebook.py`.

---

### Task 1: `roles.py` — papéis aprendidos na vitória

**Files:**
- Create: `agents/causal/roles.py`
- Test: `tests/causal/test_roles.py`

**Interfaces:**
- Consumes: `agents.causal.perception.Object` (campos `color:int, cells:frozenset, bbox:(min_row,min_col,max_row,max_col), centroid:(row,col), size:int, shape_hash:str, id`).
- Produces: `WinRoles(target_color: int, mover_size: int)` (dataclass frozen) e `learn_win_roles(pre_objs, win_objs, avatar_obj=None, max_size=100, tol=2) -> WinRoles | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/causal/test_roles.py
"""Papeis aprendidos na vitoria (spec 2026-09-05-reach-reward-roles): movel = o que
sumiu da posicao pre-vitoria (ou o avatar), pouso = estatico coberto pelo objeto novo."""
from agents.causal.perception import Object
from agents.causal.roles import WinRoles, learn_win_roles


def _obj(color, r0, c0, h, w, oid=None):
    cells = frozenset((r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w))
    return Object(color, cells, (r0, c0, r0 + h - 1, c0 + w - 1),
                  ((2 * r0 + h - 1) / 2, (2 * c0 + w - 1) / 2), h * w, f"sh{h}x{w}", oid)


def test_mover_merges_into_same_color_target():
    # tn36 L1: bloco (11,3x3) sumiu de (10,10); alvo (11,4x4) em (30,30) virou um objeto
    # maior (fundido); ponto da grade (4,4x4) reapareceu onde o bloco estava.
    pre = [_obj(11, 10, 10, 3, 3), _obj(11, 30, 30, 4, 4), _obj(2, 50, 50, 2, 2)]
    win = [_obj(11, 30, 30, 5, 5), _obj(4, 10, 10, 4, 4), _obj(2, 50, 50, 2, 2)]
    assert learn_win_roles(pre, win) == WinRoles(target_color=11, mover_size=9)


def test_mover_lands_on_other_color_target():
    # movel cor 5 (3x3) vai de (10,10) pra dentro do alvo cor 7 (5x5) em (30,30)
    pre = [_obj(5, 10, 10, 3, 3), _obj(7, 30, 30, 5, 5)]
    win = [_obj(5, 31, 31, 3, 3), _obj(7, 30, 30, 5, 5)]   # anel cor 7 tem cells a menos
    win[1] = Object(7, win[1].cells - win[0].cells, win[1].bbox, win[1].centroid, 16, "ring", None)
    assert learn_win_roles(pre, win) == WinRoles(target_color=7, mover_size=9)


def test_avatar_given_adjacent_landing():
    # avatar conhecido pousa AO LADO (gap <= 2) do alvo, sem sobrepor
    av = _obj(9, 10, 10, 2, 2, oid=7)
    pre = [av, _obj(3, 10, 40, 2, 2)]
    win = [_obj(9, 10, 37, 2, 2), _obj(3, 10, 40, 2, 2)]
    assert learn_win_roles(pre, win, avatar_obj=av) == WinRoles(target_color=3, mover_size=4)


def test_two_movers_without_avatar_is_ambiguous():
    pre = [_obj(5, 10, 10, 3, 3), _obj(6, 20, 20, 3, 3), _obj(7, 40, 40, 5, 5)]
    win = [_obj(5, 41, 41, 3, 3), _obj(6, 30, 30, 3, 3), _obj(7, 40, 40, 5, 5)]
    assert learn_win_roles(pre, win) is None


def test_no_landing_returns_none():
    pre = [_obj(5, 10, 10, 3, 3), _obj(7, 40, 40, 5, 5)]
    win = [_obj(5, 20, 20, 3, 3), _obj(7, 40, 40, 5, 5)]   # moveu pro vazio
    assert learn_win_roles(pre, win) is None


def test_big_objects_ignored():
    pre = [_obj(0, 0, 0, 64, 64), _obj(5, 10, 10, 3, 3), _obj(7, 30, 30, 5, 5)]
    win = [_obj(0, 0, 0, 60, 64), _obj(5, 31, 31, 3, 3), _obj(7, 30, 30, 5, 5)]
    assert learn_win_roles(pre, win) == WinRoles(target_color=7, mover_size=9)


def test_bad_input_returns_none():
    assert learn_win_roles(None, [1, 2]) is None
    assert learn_win_roles([], []) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/causal/test_roles.py -q`
Expected: FAIL com `ModuleNotFoundError: agents.causal.roles`.

- [ ] **Step 3: Write minimal implementation**

```python
# agents/causal/roles.py
"""Papéis aprendidos na vitória (spec 2026-09-05-reach-reward-roles).

No level-up comparamos a última cena de decisão com a cena de vitória:
- MÓVEL = o objeto que sumiu da posição pré-vitória (ou o avatar do MovementModel);
- POUSO = objeto estático coberto (ou encostado) pelo objeto novo onde o móvel parou.
Guardamos a COR do pouso (estável entre níveis) e o TAMANHO do móvel (a cor do móvel
pode mudar de nível pra nível, ex. tn36 L1 cor 11 → L2 cor 4). Puro, sem estado,
exception-safe (qualquer problema → None)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WinRoles:
    target_color: int
    mover_size: int


def _bbox_gap(a, b) -> int:
    """Distância (Manhattan) entre bboxes; 0 se intersectam."""
    dr = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dc = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return dr + dc


def _match(o, objs, tol) -> bool:
    for p in objs:
        if (p.color == o.color and abs(p.size - o.size) <= 2
                and abs(p.centroid[0] - o.centroid[0]) + abs(p.centroid[1] - o.centroid[1]) <= tol):
            return True
    return False


def learn_win_roles(pre_objs, win_objs, avatar_obj=None, max_size=100, tol=2):
    try:
        pre = [o for o in pre_objs if o.size <= max_size]
        win = [o for o in win_objs if o.size <= max_size]
        if not pre or not win:
            return None
        unmatched_pre = [o for o in pre if not _match(o, win, tol)]
        new_win = [w for w in win if not _match(w, pre, tol)]
        if avatar_obj is not None:
            mover = avatar_obj
        else:
            # Móvel: sumiu E nenhum objeto da mesma cor cobre a posição antiga (o pouso,
            # ao contrário, continua com a sua cor no lugar mesmo se mudou de tamanho).
            cands = [o for o in unmatched_pre
                     if not any(w.color == o.color and _bbox_gap(w.bbox, o.bbox) == 0 for w in win)]
            if len(cands) != 1:
                return None
            mover = cands[0]
        others = [o for o in pre if o is not mover]
        # Destino do móvel: objeto novo da mesma cor/tamanho (moveu intacto) ...
        dest = [w for w in new_win if w.color == mover.color and abs(w.size - mover.size) <= 2]
        landing = []
        if dest:
            d = dest[0]
            landing = [o for o in others if _bbox_gap(o.bbox, d.bbox) == 0]
            if not landing:      # pousou encostado (avatar ao lado do alvo)
                landing = [o for o in others if _bbox_gap(o.bbox, d.bbox) <= tol]
        if not landing:
            # ... ou fundiu-se com o pouso (mesma cor): pouso = estático coberto por um
            # objeto novo MAIOR que ele.
            landing = [o for o in others
                       if any(_bbox_gap(o.bbox, w.bbox) == 0 and w.size > o.size for w in new_win)]
        if not landing:
            return None
        best = min(landing, key=lambda o: abs(o.size - mover.size))
        return WinRoles(target_color=int(best.color), mover_size=int(mover.size))
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/causal/test_roles.py -q`
Expected: 7 passed. Se `test_mover_merges_into_same_color_target` falhar porque o ponto da grade (4,4x4) novo cobre a posição antiga do bloco: ele é de cor 4 ≠ 11, então não descarta o móvel — confira `_bbox_gap`.

- [ ] **Step 5: Commit**

```bash
git add agents/causal/roles.py tests/causal/test_roles.py
git commit -m "feat(reach): roles.py — papeis (movel, pouso) aprendidos na cena de vitoria"
```

---

### Task 2: `grounded_reach_reward_fn` em `goals.py`

**Files:**
- Modify: `agents/causal/goals.py` (inserir logo após `grounded_reward_fn`, ~linha 93)
- Test: `tests/causal/test_grounded_reach.py`

**Interfaces:**
- Consumes: formato de `state` = lista de `(shape_hash, {"x","y","color","shape","size"})` (ver `agent._obj_state`).
- Produces: `grounded_reach_reward_fn(mover_color:int, mover_size:int, target_color:int, max_size:int=100) -> reward_fn`, `reward_fn(state) -> (float, bool)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/causal/test_grounded_reach.py
"""Reward de alcance: -dist(movel, objeto de target_color mais parecido em tamanho),
excluindo o proprio movel; goal em dist 0; sem par -> (0.0, False)."""
from agents.causal.goals import grounded_reach_reward_fn


def _st(*objs):
    return [(f"s{i}", {"x": x, "y": y, "color": c, "shape": f"s{i}", "size": s})
            for i, (c, s, x, y) in enumerate(objs)]


def test_distance_and_goal():
    fn = grounded_reach_reward_fn(4, 14, 11)
    assert fn(_st((4, 14, 10, 10), (11, 14, 13, 14))) == (-7.0, False)
    assert fn(_st((4, 14, 13, 14), (11, 14, 13, 14))) == (0.0, True)


def test_target_picked_by_size_closest_to_mover():
    fn = grounded_reach_reward_fn(4, 14, 11)
    st = _st((4, 14, 0, 0), (11, 6, 1, 0), (11, 14, 20, 0))   # o de tamanho 6 esta perto
    assert fn(st) == (-20.0, False)


def test_same_color_excludes_mover_itself():
    fn = grounded_reach_reward_fn(11, 14, 11)
    st = _st((11, 14, 0, 0), (11, 16, 5, 5))
    assert fn(st) == (-10.0, False)


def test_missing_mover_or_target():
    fn = grounded_reach_reward_fn(4, 14, 11)
    assert fn(_st((4, 14, 0, 0))) == (0.0, False)
    assert fn(_st((11, 14, 0, 0))) == (0.0, False)
    assert fn(_st((4, 14, 0, 0), (11, 500, 9, 9))) == (0.0, False)   # alvo grande ignorado
    assert fn("lixo") == (0.0, False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/causal/test_grounded_reach.py -q`
Expected: FAIL com `ImportError: cannot import name 'grounded_reach_reward_fn'`.

- [ ] **Step 3: Write minimal implementation** (após `grounded_reward_fn`)

```python
def grounded_reach_reward_fn(mover_color, mover_size, target_color, max_size=100):
    """Reward de ALCANCE com papéis (spec 2026-09-05-reach-reward-roles): -manhattan do
    móvel (cor+tamanho aprendidos pelo MovementModel neste nível) ao objeto de
    `target_color` (cor do pouso na vitória anterior) de tamanho mais próximo do móvel,
    EXCLUINDO o próprio móvel (as cores podem coincidir) e objetos grandes (fundo/HUD).
    goal_flag em dist==0. Exception-safe -> (0.0, False)."""
    def reward_function(state):
        try:
            objs = [a for _, a in state if a.get("size", 10 ** 9) <= max_size]
            movers = [o for o in objs if o.get("color") == mover_color]
            if not movers:
                return (0.0, False)
            m = min(movers, key=lambda o: abs(o.get("size", 0) - mover_size))
            targets = [o for o in objs if o.get("color") == target_color and o is not m]
            if not targets:
                return (0.0, False)
            t = min(targets, key=lambda o: abs(o.get("size", 0) - mover_size))
            d = abs(m["x"] - t["x"]) + abs(m["y"] - t["y"])
            return (-float(d), d == 0)
        except Exception:
            return (0.0, False)
    return reward_function
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/causal/test_grounded_reach.py tests/causal/test_goals.py -q`
Expected: todos passam.

- [ ] **Step 5: Commit**

```bash
git add agents/causal/goals.py tests/causal/test_grounded_reach.py
git commit -m "feat(reach): grounded_reach_reward_fn — distancia movel->cor do pouso"
```

---

### Task 3: Wiring em `agent.py` sob `CAUSAL_REACH`

**Files:**
- Modify: `agents/causal/agent.py` — imports (~linha 33), init (~linha 392-394, bloco winreward), level-up (~linha 559-565, onde `select_win_reward` roda), fecha-loop (~linha 585, junto de `_track_rprog`), `choose_action` após `_grounded_reward_step` (~linha 684), `phase2_stats` (~linha 1606).
- Test: `tests/causal/test_agent_reach.py`

**Interfaces:**
- Consumes: `learn_win_roles`, `WinRoles` (Task 1); `grounded_reach_reward_fn` (Task 2); existentes: `self._move.avatar_id()`, `self._prev_scene`, `self._buffer`, `self._grounded_states(scene)`, `accept_reward_fn`, `value_fn_from_reward`, `_background_colors(scene)`, `self._rprog`, `self._win_reward`, `self._reward_rejected`.
- Produces: atributos `self._reach_on: bool`, `self._win_roles: WinRoles|None`, `self._flat_steps: int`, `self._flat_steps_max: int`, `self._reach_src: str|None`; método `self._maybe_adopt_reach(scene) -> bool`; chaves `win_roles`, `reach_src`, `flat_steps_max` em `phase2_stats()`. Constante `REACH_FLAT_K = 8` (módulo).

- [ ] **Step 1: Write the failing tests**

```python
# tests/causal/test_agent_reach.py
"""Wiring CAUSAL_REACH (spec 2026-09-05-reach-reward-roles): adota a reward de alcance
so quando (a) ha papeis da vitoria, (b) o MovementModel conhece o movel, (c) a reward
ativa ficou plana por REACH_FLAT_K passos com efeito visivel. Flag off: nada existe."""
from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent, REACH_FLAT_K
from agents.causal.roles import WinRoles

CLICK = [GameAction.ACTION6]


class _Frame:
    def __init__(self, frame, levels=1):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = levels
        self.available_actions = CLICK
        self.full_reset = False


def _grid(mover_x, mover_color=3):
    """Fundo 0; alvo cor 7 (3x3) fixo em cols 50..52; movel 3x3 em cols mover_x..+2."""
    g = [[0] * 64 for _ in range(64)]
    for r in range(30, 33):
        for c in range(50, 53):
            g[r][c] = 7
        for c in range(mover_x, mover_x + 3):
            g[r][c] = mover_color
    return g


def _agent(monkeypatch, **env):
    env.setdefault("CAUSAL_LLM", "0")
    env.setdefault("CAUSAL_GK", "0")
    env.setdefault("CAUSAL_GROUNDED", "1")
    env.setdefault("CAUSAL_RPROG", "1")
    env.setdefault("CAUSAL_WINREWARD", "1")
    env.setdefault("CAUSAL_MAX_ACTIONS", "10000")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.game_id = "zzzz"
    a.MAX_ACTIONS = 10000
    a._init_causal_state()
    return a


def _flat_reward(_state):
    return (0.0, False)


def _prime(a, roles=WinRoles(target_color=7, mover_size=9)):
    """Estado pos-level-up: reward validada (plana) ativa + papeis conhecidos."""
    a._win_roles = roles
    a._win_reward = ("multi-align", _flat_reward, 0.9)
    a._reward_fn, a._reward_src = _flat_reward, "win:multi-align(rho=0.90)"


def _run(a, n=REACH_FLAT_K + 4):
    """Movel anda 1 col por passo (efeito visivel), reward plana o tempo todo."""
    for i in range(n):
        a.choose_action([], _Frame([_grid(5 + i)]))
        a.action_counter += 1


def test_adopts_reach_after_flat_steps(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_REACH="1")
    _prime(a)
    _run(a)
    assert a._reach_src == "reach:cor3#9->cor7"
    assert a._reward_src == a._reach_src
    assert a._win_reward[0] == "reach"
    assert a._flat_steps == 0
    st = a.phase2_stats()
    assert st["reach_src"] == a._reach_src
    assert st["win_roles"] == "cor7#9"
    assert st["flat_steps_max"] >= REACH_FLAT_K


def test_reach_reward_has_gradient_after_adoption(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_REACH="1")
    _prime(a)
    _run(a)
    fn = a._reward_fn
    near = [("s", {"x": 49, "y": 31, "color": 3, "shape": "s", "size": 9}),
            ("t", {"x": 51, "y": 31, "color": 7, "shape": "t", "size": 9})]
    far = [("s", {"x": 10, "y": 31, "color": 3, "shape": "s", "size": 9}),
           ("t", {"x": 51, "y": 31, "color": 7, "shape": "t", "size": 9})]
    assert fn(near)[0] > fn(far)[0]


def test_no_adoption_before_k_steps(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_REACH="1")
    _prime(a)
    _run(a, n=REACH_FLAT_K - 2)
    assert a._reach_src is None and a._reward_src.startswith("win:multi-align")


def test_no_adoption_without_roles(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_REACH="1")
    _prime(a, roles=None)
    _run(a)
    assert a._reach_src is None and a._reward_src.startswith("win:multi-align")


def test_no_adoption_when_reward_moves(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_REACH="1")
    _prime(a)
    def moving(state):        # varia com a posicao do movel -> nao e plana
        xs = [o["x"] for _, o in state if o["color"] == 3]
        return (float(xs[0]) if xs else 0.0, False)
    a._win_reward = ("multi-align", moving, 0.9)
    a._reward_fn = moving
    _run(a)
    assert a._reach_src is None and a._flat_steps_max == 0


def test_flag_off_is_inert(monkeypatch):
    a = _agent(monkeypatch)
    _prime(a)
    _run(a)
    assert a._reach_src is None and a._win_roles is not None   # so o wiring nao roda
    assert a._reward_src.startswith("win:multi-align")
    assert a.phase2_stats()["reach_src"] is None


def test_rejected_by_accept_counts_and_retries(monkeypatch):
    """Alvo cor 7 AUSENTE: reach devolve (0.0,False) constante -> accept_reward_fn
    rejeita; conta rejeicao e nao adota."""
    a = _agent(monkeypatch, CAUSAL_REACH="1")
    _prime(a)
    before = a._reward_rejected
    for i in range(REACH_FLAT_K + 4):
        g = [[0] * 64 for _ in range(64)]
        for r in range(30, 33):
            for c in range(5 + i, 8 + i):
                g[r][c] = 3
        a.choose_action([], _Frame([g]))
        a.action_counter += 1
    assert a._reach_src is None
    assert a._reward_rejected > before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/causal/test_agent_reach.py -q`
Expected: FAIL com `ImportError: cannot import name 'REACH_FLAT_K'`.

- [ ] **Step 3: Implement — imports e constante** (topo de `agent.py`, junto dos outros `from .` e constantes)

```python
from .roles import WinRoles, learn_win_roles
from .goals import grounded_reach_reward_fn   # juntar ao import existente de .goals

REACH_FLAT_K = 8   # passos com efeito visivel sem a reward mudar -> adota a reach
```

- [ ] **Step 4: Implement — estado no `_init_causal_state`** (logo após `self._win_reward = None`)

```python
        # Lever "reward de alcance com papeis" (spec 2026-09-05-reach-reward-roles):
        # quando a reward validada fica PLANA no nivel seguinte, mede a distancia do
        # movel (MovementModel) ao objeto da cor onde o movel pousou na vitoria.
        self._reach_on = os.environ.get("CAUSAL_REACH", "0") != "0"
        self._win_roles = None            # WinRoles | None (persiste entre niveis)
        self._flat_steps = 0              # passos com efeito e reward igual (seguidos)
        self._flat_steps_max = 0          # diag
        self._reach_src = None            # "reach:cor{m}#{s}->cor{t}" quando adotada
```

Confirme que `_init_causal_state` é chamado no `full_reset` (linha ~479-483) — o estado acima reseta junto.

- [ ] **Step 5: Implement — papéis no level-up** (dentro do bloco `if level_up:`, logo após `win_scene = parse(wg, hud_mask=self._hud.mask())`)

```python
                if self._reach_on and self._prev_scene is not None:
                    aid = self._move.avatar_id()
                    av = next((o for o in self._prev_scene.objects if o.id == aid), None) \
                        if aid is not None else None
                    roles = learn_win_roles(self._prev_scene.objects, win_scene.objects, av)
                    if roles is not None:
                        self._win_roles = roles
                    self._flat_steps = 0
                    self._reach_src = None    # a reach e por-nivel (movel muda de cor)
```

- [ ] **Step 6: Implement — detector de planura no fecha-loop** (logo após `self._track_rprog(scene)`; `actual` e `self._last_pixel_delta` já existem ali)

```python
                if self._reach_on and self._reward_fn is not None and self._prev_scene is not None:
                    self._track_flat(scene, actual)
```

e o método (perto de `_track_rprog`):

```python
    def _track_flat(self, scene, actual) -> None:
        """Conta passos SEGUIDOS com efeito visivel em que a reward ativa nao mudou.
        So decisao->decisao. Mudou -> zera."""
        visible = actual.kind != "none" or self._last_pixel_delta >= 5
        if not visible:
            return
        vf = value_fn_from_reward(self._reward_fn)
        vb = vf([(o.shape_hash, _obj_state(o)) for o in self._prev_scene.objects])
        va = vf([(o.shape_hash, _obj_state(o)) for o in scene.objects])
        if math.isfinite(vb) and math.isfinite(va) and vb == va:
            self._flat_steps += 1
            self._flat_steps_max = max(self._flat_steps_max, self._flat_steps)
        else:
            self._flat_steps = 0
```

- [ ] **Step 7: Implement — adoção** (em `choose_action`, logo após o bloco `if self._grounded: self._grounded_reward_step(...)`)

```python
        if self._reach_on:
            self._maybe_adopt_reach(scene)
```

e o método:

```python
    def _maybe_adopt_reach(self, scene) -> bool:
        """Adota a reward de alcance quando: papeis conhecidos, reward ativa plana ha
        REACH_FLAT_K passos com efeito, movel conhecido na cena atual (fora do fundo) e
        a reward passa no accept_reward_fn (tem gradiente nos estados reais)."""
        if (self._reach_src is not None or self._win_roles is None
                or self._flat_steps < REACH_FLAT_K):
            return False
        aid = self._move.avatar_id()
        av = next((o for o in scene.objects if o.id == aid), None) if aid is not None else None
        if av is None or av.color in _background_colors(scene):
            return False
        fn = grounded_reach_reward_fn(av.color, av.size, self._win_roles.target_color)
        ok, _reason = accept_reward_fn(fn, self._grounded_states(scene))
        self._flat_steps = 0
        if not ok:
            self._reward_rejected += 1
            return False
        self._reach_src = f"reach:cor{av.color}#{av.size}->cor{self._win_roles.target_color}"
        self._reward_fn, self._reward_src = fn, self._reach_src
        self._win_reward = ("reach", fn, 0.0)      # mantem o rprog sem cap acima do 2phase
        self._rprog.clear()
        return True
```

- [ ] **Step 8: Implement — telemetria** (em `phase2_stats`, após `"win_rho"`)

```python
            "win_roles": None if self._win_roles is None
                         else f"cor{self._win_roles.target_color}#{self._win_roles.mover_size}",
            "reach_src": self._reach_src,
            "flat_steps_max": self._flat_steps_max,
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest tests/causal/test_agent_reach.py -q`
Expected: 7 passed. Se `test_adopts_reach_after_flat_steps` falhar por `avatar_id() is None`: o `MovementModel.observe` só conta o móvel quando `_moved_group` acha 1 mover rígido — confira que o grid de teste move só o bloco (3x3 idêntico) e que a cor 3 não é fundo (`_background_colors` = cor 0 + cor com >50% da área). Se falhar no `accept_reward_fn` por "poucos estados", o buffer tem ≥3 transições após 12 passos — confira `self._buffer.append` no fecha-loop.

Run: `uv run pytest tests/causal tests/kaggle -x -q`
Expected: 519 + 7 + 4 + 7 = 537 passed (caminho off inalterado).

- [ ] **Step 10: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_reach.py
git commit -m "feat(reach): CAUSAL_REACH — adota reward de alcance quando a validada fica plana"
```

---

### Task 4: Flag nos builders + `roles.py` embarcado

**Files:**
- Modify: `kaggle/build_notebook.py` (`MODULES` linha 10-15; `ENV` após `"CAUSAL_WINREWARD=1\n"` ~linha 90)
- Modify: `kaggle/build_offline_notebook.py` (`OFFLINE_ENV` após `"CAUSAL_WINREWARD=1\n"` ~linha 36)
- Test: `tests/kaggle/test_build_notebook.py`, `tests/kaggle/test_build_offline_notebook.py`

**Interfaces:**
- Consumes: `read_sources`/`build_notebook` existentes; `test_modules_cover_all_relative_imports` (já pega `roles.py` faltando).
- Produces: `CAUSAL_REACH=1` em `ENV` e `OFFLINE_ENV`; `"roles.py"` em `MODULES`.

- [ ] **Step 1: Write the failing tests** (acrescentar ao fim de cada arquivo)

```python
# tests/kaggle/test_build_notebook.py
def test_env_has_reach():
    import kaggle.build_notebook as b
    assert "CAUSAL_REACH=1\n" in b.ENV
    assert "roles.py" in b.MODULES
```

```python
# tests/kaggle/test_build_offline_notebook.py
def test_offline_env_has_reach():
    import kaggle.build_offline_notebook as b
    assert "CAUSAL_REACH=1\n" in b.OFFLINE_ENV
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/kaggle -q`
Expected: os 2 novos falham; `test_modules_cover_all_relative_imports` também falha (`agent.py -> .roles`).

- [ ] **Step 3: Implement**

`kaggle/build_notebook.py`: acrescentar `"roles.py"` ao fim de `MODULES` e, após a linha `"CAUSAL_WINREWARD=1\n"`, a linha
`"CAUSAL_REACH=1\n"        # reward de alcance com papeis da vitoria quando a validada fica plana`.
`kaggle/build_offline_notebook.py`: mesma linha após `"CAUSAL_WINREWARD=1\n"` em `OFFLINE_ENV`.

- [ ] **Step 4: Regenerar e verificar embed**

```bash
uv run python kaggle/build_notebook.py && uv run python kaggle/build_offline_notebook.py
uv run python -c "
import json,base64
for nb in ('kaggle/submission.ipynb','kaggle/offline.ipynb'):
    src=''.join(json.load(open(nb))['cells'][1]['source'])
    assert 'agents/causal/roles.py' in src and 'CAUSAL_REACH=1' in src, nb
print('embed ok')"
uv run pytest tests/causal tests/kaggle -x -q
```
Expected: `embed ok`; 539 passed.

- [ ] **Step 5: Commit**

```bash
git add kaggle/build_notebook.py kaggle/build_offline_notebook.py kaggle/submission.ipynb kaggle/offline.ipynb tests/kaggle/test_build_notebook.py tests/kaggle/test_build_offline_notebook.py
git commit -m "feat(reach): CAUSAL_REACH=1 nos builders, roles.py embarcado; notebooks regenerados"
```

---

### Task 5: Validação real — probe no recording e run cego local

**Files:**
- Create: `analysis/reach_probe.py`

**Interfaces:**
- Consumes: `perception.parse/to_grid/win_grid`, `HudMask`, `learn_win_roles`, `grounded_reach_reward_fn`, `_obj_state`.
- Produces: impressão dos papéis do L1 e dos valores da reach no L2 (distintos + nas execuções multi-grid).

- [ ] **Step 1: Escrever o probe**

```python
# analysis/reach_probe.py — valida a spec 2026-09-05-reach-reward-roles num recording.
# uso: uv run python analysis/reach_probe.py recordings/tn36-...recording.jsonl
import json, sys
from agents.causal import perception as P
from agents.causal.hud import HudMask
from agents.causal.roles import learn_win_roles
from agents.causal.goals import grounded_reach_reward_fn, value_fn_from_reward
from agents.causal.agent import _obj_state

rows = [json.loads(l)["data"] for l in open(sys.argv[1])]
hud, prev, prev_scene, roles, vf = HudMask(), None, None, None, None
vals, runs, level = [], [], 0
for i, r in enumerate(rows):
    g = P.to_grid(r["frame"])
    if prev is not None:
        hud.update(prev, g)
    prev = g
    if r["levels_completed"] > level:          # level-up: aprende papeis na vitoria
        level = r["levels_completed"]
        win_scene = P.parse(P.win_grid(r["frame"]), hud_mask=hud.mask())
        roles = learn_win_roles(prev_scene.objects, win_scene.objects) if prev_scene else None
        print(f"level-up @{i}: roles={roles}")
        vf = None
    scene = P.parse([r["frame"][-1]], hud_mask=hud.mask())
    if roles is not None and vf is None:
        # movel do nivel novo: menor objeto pequeno com tamanho == mover_size (proxy do
        # MovementModel, que so existe no agente ao vivo)
        cand = [o for o in scene.objects if o.size <= 100 and abs(o.size - roles.mover_size) <= 2
                and o.color != roles.target_color]
        if cand:
            m = cand[0]
            vf = value_fn_from_reward(grounded_reach_reward_fn(m.color, m.size, roles.target_color))
            print(f"movel proxy: cor{m.color}#{m.size}")
    if vf is not None:
        v = vf([(o.shape_hash, _obj_state(o)) for o in scene.objects])
        vals.append(v)
        if len(r["frame"]) > 1:
            runs.append((i, v))
    prev_scene = scene
print("distintos:", len(set(vals)), "| nas execucoes:", runs[:20])
```

- [ ] **Step 2: Rodar o probe no recording do tn36**

Run: `uv run python analysis/reach_probe.py recordings/tn36-ef4dde99.causalobjectagent.4543083f-ae1e-40d3-97ae-42353a57ea2c.recording.jsonl`
Expected: `roles=WinRoles(target_color=11, mover_size=14)`, `movel proxy: cor4#14`, `distintos >= 3`, e valores diferentes entre execuções (hoje a multi-align dá 1 distinto). Se `roles=None`: imprima `unmatched_pre`/`new_win` e ajuste `tol` — o bloco do L1 é (11,14)@(14,31) e o alvo (11,16)@(35,31), fundidos em (11,30) na vitória.

- [ ] **Step 3: Run cego local nos 25** (API pública, ~35 min; requer `ARC_API_KEY` no `.env`)

```bash
CAUSAL_LLM=0 CAUSAL_GK=0 CAUSAL_GROUNDED=1 CAUSAL_RPROG=1 CAUSAL_COVER=1 CAUSAL_CLICKMAP=1 \
CAUSAL_WINREWARD=1 CAUSAL_FIX=1 CAUSAL_REACH=1 CAUSAL_MAX_ACTIONS=200 \
uv run main.py --agent=causalobject 2>&1 | tee /tmp/claude-1000/-home-lkenzo-projetos-safe/06327d1a-970f-4211-baa9-b5a375c2db44/scratchpad/reach_blind.log
grep -E "DONE|phase2 stats" /tmp/claude-1000/-home-lkenzo-projetos-safe/06327d1a-970f-4211-baa9-b5a375c2db44/scratchpad/reach_blind.log | grep -oE "^.*DONE.*|'game_id': '[a-z0-9]+'|'reach_src': '[^']*'|'win_roles': '[^']*'|'rprog_fires': [0-9]+"
```
Expected: total `>= 5L` (vc33 2, tn36 1, lp85 1, lf52 1 preservados); tn36 com `reach_src` preenchido e `rprog_fires > 0` no L2. Se o placar cair: comparar por jogo qual `reward_src` mudou e ajustar (a reach só deve entrar onde a validada estava plana).

- [ ] **Step 4: Commit + log**

```bash
git add analysis/reach_probe.py
git commit -m "test(reach): probe de validacao da reward de alcance sobre recordings"
```
Registrar no `CLAUDE.md` (raiz) o resultado do probe e do run cego (placar por jogo, `reach_src`/`win_roles` do tn36).
