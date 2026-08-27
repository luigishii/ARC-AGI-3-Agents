# CausalObjectAgent v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Um agente ARC-AGI-3 que constrói um modelo causal objeto-cêntrico interpretável (numpy puro) e decide por ganho de informação + exploit de progresso, sem GPU/LLM.

**Architecture:** Quatro componentes isolados — `Perception` (grade→objetos), `CausalModel` (transições simbólicas ação→efeito), `Policy` (score de 4 termos + FSM EXPLORAR/EXPLOITAR) e `Instrumentation` (JSONL) — orquestrados por `CausalObjectAgent(Agent)` no `choose_action`.

**Tech Stack:** Python 3.12, numpy, pytest. Harness oficial `ARC-AGI-3-Agents` (classe base `agents.agent.Agent`, tipos `arcengine.FrameData/GameAction/GameState`). Rodar via `uv`.

## Global Constraints

- **Sem GPU, sem LLM, sem internet em runtime.** Só numpy + stdlib. Nada de scipy/torch (offline eval).
- **Orçamento ~80 ações/jogo** (`Agent.MAX_ACTIONS` default 80). Eficiência conta no score.
- **Grade:** `latest_frame.frame` é `list[list[list[int]]]` (pilha de camadas 64×64, valores 0–15). A grade atual é `frame[-1]`. No fim de nível a pilha pode ter >1 camada.
- **Ação:** decidir x,y só quando `action.is_complex()` é True; usar `available_actions` dinamicamente (não hardcodar quais ações existem). RESET quando `state ∈ {NOT_PLAYED, GAME_OVER}` ou `full_reset`.
- **Determinismo:** Policy é determinística com ε pequeno; seed fixa nos testes.
- **Logar a ação nós mesmos** (id+x,y+reasoning) — o `action_input` das gravações oficiais é placeholder.
- Rodar testes: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/ -v`.

---

### Task 0: Scaffold do pacote + registro do agente (stub end-to-end)

**Files:**
- Create: `agents/causal/__init__.py`
- Create: `agents/causal/agent.py`
- Modify: `agents/__init__.py` (adicionar `"causalobject"` ao `AVAILABLE_AGENTS`)
- Test: `tests/causal/__init__.py`, `tests/causal/test_agent_smoke.py`

**Interfaces:**
- Consumes: `agents.agent.Agent`, `arcengine.{FrameData, GameAction, GameState}`.
- Produces: `CausalObjectAgent(Agent)` com `choose_action(frames, latest_frame) -> GameAction` e `is_done(frames, latest_frame) -> bool`. Registrado como `"causalobject"`.

- [ ] **Step 1: Escrever o teste de fumaça (falha)**

```python
# tests/causal/test_agent_smoke.py
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_agent_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError: agents.causal.agent`).

- [ ] **Step 3: Implementar o stub mínimo**

```python
# agents/causal/agent.py
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
        actions = latest_frame.available_actions or [GameAction.ACTION1]
        action = actions[0]
        action.reasoning = {"stage": "stub"}
        return action
```

```python
# agents/causal/__init__.py
from .agent import CausalObjectAgent

__all__ = ["CausalObjectAgent"]
```

```python
# tests/causal/__init__.py  (arquivo vazio)
```

- [ ] **Step 4: Registrar o agente**

Em `agents/__init__.py`, adicionar o import e a entrada no dict `AVAILABLE_AGENTS`:

```python
from .causal import CausalObjectAgent
# ...
AVAILABLE_AGENTS["causalobject"] = CausalObjectAgent
```

- [ ] **Step 5: Rodar e ver passar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_agent_smoke.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add agents/causal/ agents/__init__.py tests/causal/
git commit -m "feat(causal): scaffold CausalObjectAgent + registro causalobject"
```

---

### Task 1: `Perception` — segmentação em objetos (`parse`)

**Files:**
- Create: `agents/causal/perception.py`
- Test: `tests/causal/test_perception.py`

**Interfaces:**
- Consumes: numpy; grade `list[list[int]]` (uma camada 64×64) ou `list[list[list[int]]]` (pilha — usa a última).
- Produces:
  - `@dataclass(frozen=True) class Object: color:int; cells:frozenset[tuple[int,int]]; bbox:tuple[int,int,int,int]; centroid:tuple[float,float]; size:int; shape_hash:str; id:int|None=None`
  - `@dataclass class Scene: objects:list[Object]; grid:np.ndarray`
  - `def parse(frame) -> Scene` — connected-components 4-vizinhança por cor; **ignora a cor de fundo** (a mais frequente na grade).
  - `def object_at(scene:Scene, x:int, y:int) -> Object|None`

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_perception.py
import numpy as np
from agents.causal.perception import parse, object_at


def test_two_objects_ignoring_background():
    grid = np.zeros((5, 5), dtype=int)
    grid[0, 0] = 3            # objeto A (1 célula)
    grid[2:4, 2:4] = 7        # objeto B (bloco 2x2)
    scene = parse(grid.tolist())
    assert len(scene.objects) == 2
    sizes = sorted(o.size for o in scene.objects)
    assert sizes == [1, 4]
    colors = sorted(o.color for o in scene.objects)
    assert colors == [3, 7]


def test_bbox_and_centroid():
    grid = np.zeros((5, 5), dtype=int)
    grid[1:3, 1:4] = 5        # bloco 2 linhas x 3 colunas
    scene = parse(grid.tolist())
    (o,) = scene.objects
    assert o.bbox == (1, 1, 2, 3)          # (min_row, min_col, max_row, max_col)
    assert o.centroid == (1.5, 2.0)        # (row, col) médio
    assert o.size == 6


def test_stack_uses_last_layer():
    layer0 = np.zeros((3, 3), dtype=int).tolist()
    layer1 = np.zeros((3, 3), dtype=int)
    layer1[0, 0] = 9
    scene = parse([layer0, layer1.tolist()])
    assert len(scene.objects) == 1 and scene.objects[0].color == 9


def test_object_at():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 4
    scene = parse(grid.tolist())
    assert object_at(scene, x=2, y=2).color == 4     # x=col, y=row
    assert object_at(scene, x=0, y=0) is None
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_perception.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar `perception.py`**

```python
# agents/causal/perception.py
from __future__ import annotations

import hashlib
import itertools
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

_id_counter = itertools.count(1)


@dataclass(frozen=True)
class Object:
    color: int
    cells: frozenset          # frozenset[tuple[int,int]] em (row,col)
    bbox: tuple               # (min_row, min_col, max_row, max_col)
    centroid: tuple           # (row, col) float
    size: int
    shape_hash: str
    id: int | None = None


@dataclass
class Scene:
    objects: list
    grid: np.ndarray = field(repr=False)


def _to_grid(frame) -> np.ndarray:
    arr = np.array(frame)
    while arr.ndim > 2:        # pilha de camadas -> última
        arr = arr[-1]
    return arr.astype(int)


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _shape_hash(cells: frozenset) -> str:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return hashlib.md5(str(norm).encode()).hexdigest()[:8]


def parse(frame) -> Scene:
    grid = _to_grid(frame)
    bg = _background_color(grid)
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            cells = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            cset = frozenset(cells)
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
            objects.append(
                Object(color, cset, bbox, centroid, len(cells), _shape_hash(cset))
            )
    return Scene(objects=objects, grid=grid)


def object_at(scene: Scene, x: int, y: int):
    for o in scene.objects:
        if (y, x) in o.cells:      # x=col, y=row
            return o
    return None
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_perception.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/perception.py tests/causal/test_perception.py
git commit -m "feat(causal): Perception.parse (connected-components objeto-centrico)"
```

---

### Task 2: `Perception` — identidade persistente (`match_objects`)

**Files:**
- Modify: `agents/causal/perception.py`
- Test: `tests/causal/test_perception_match.py`

**Interfaces:**
- Consumes: `Object`, `Scene`, `replace`, `_id_counter` (já definidos na Task 1).
- Produces: `def match_objects(prev:Scene|None, curr:Scene) -> Scene` — atribui `id` estável aos objetos de `curr` casando com `prev` por (shape_hash + color) e menor distância de centroide; objetos novos recebem ids frescos de `_id_counter`. Retorna uma nova `Scene` com objetos re-`id`ados (via `dataclasses.replace`).

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_perception_match.py
import numpy as np
from agents.causal.perception import parse, match_objects


def _grid_with(pos, color=3, n=5):
    g = np.zeros((n, n), dtype=int)
    g[pos] = color
    return g.tolist()


def test_same_object_keeps_id_after_move():
    prev = match_objects(None, parse(_grid_with((1, 1))))
    id0 = prev.objects[0].id
    curr = match_objects(prev, parse(_grid_with((1, 2))))     # mesmo objeto, moveu 1 coluna
    assert curr.objects[0].id == id0


def test_new_object_gets_fresh_id():
    prev = match_objects(None, parse(_grid_with((1, 1))))
    g = np.zeros((5, 5), dtype=int)
    g[1, 1] = 3
    g[4, 4] = 7                                # objeto novo, cor diferente
    curr = match_objects(prev, parse(g.tolist()))
    ids = sorted(o.id for o in curr.objects)
    assert len(set(ids)) == 2                  # dois ids distintos
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_perception_match.py -v`
Expected: FAIL (`cannot import name 'match_objects'`).

- [ ] **Step 3: Implementar `match_objects`**

Adicionar em `perception.py` (o import de `replace` e `_id_counter` já vêm da Task 1):

```python
def match_objects(prev, curr) -> Scene:
    prev_objs = list(prev.objects) if prev is not None else []
    used = set()
    new_objs = []
    for o in curr.objects:
        best = None
        best_d = None
        for p in prev_objs:
            if p.id in used or p.color != o.color or p.shape_hash != o.shape_hash:
                continue
            d = (p.centroid[0] - o.centroid[0]) ** 2 + (p.centroid[1] - o.centroid[1]) ** 2
            if best_d is None or d < best_d:
                best, best_d = p, d
        if best is not None:
            used.add(best.id)
            new_objs.append(replace(o, id=best.id))
        else:
            new_objs.append(replace(o, id=next(_id_counter)))
    return Scene(objects=new_objs, grid=curr.grid)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_perception_match.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/perception.py tests/causal/test_perception_match.py
git commit -m "feat(causal): Perception.match_objects (identidade persistente entre frames)"
```

---

### Task 3: `CausalModel` — diff de efeito entre cenas (`compute_effect`)

**Files:**
- Create: `agents/causal/causal_model.py`
- Test: `tests/causal/test_effect.py`

**Interfaces:**
- Consumes: `Scene`/`Object` (objetos já com `id` via `match_objects`).
- Produces:
  - `Effect = namedtuple("Effect", "kind detail")` com `kind ∈ {"none","moved","appeared","disappeared","recolored","structural"}`.
  - `def compute_effect(prev:Scene, curr:Scene) -> Effect` — compara por `id`: mesmo id com centroide deslocado → `moved` (detail=`(dr,dc)` arredondado); id que sumiu → `disappeared`; id novo → `appeared`; mesma posição e cor diferente → `recolored`; nenhuma diferença → `none`; >1 mudança → `structural`.

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_effect.py
import numpy as np
from agents.causal.perception import parse, match_objects
from agents.causal.causal_model import compute_effect


def _g(pos, color=3, n=6):
    g = np.zeros((n, n), dtype=int)
    g[pos] = color
    return g.tolist()


def _scene(prev, grid):
    return match_objects(prev, parse(grid))


def test_none_effect():
    s0 = _scene(None, _g((1, 1)))
    s1 = _scene(s0, _g((1, 1)))
    assert compute_effect(s0, s1).kind == "none"


def test_moved_effect():
    s0 = _scene(None, _g((1, 1)))
    s1 = _scene(s0, _g((1, 3)))
    e = compute_effect(s0, s1)
    assert e.kind == "moved" and e.detail == (0, 2)


def test_disappeared_effect():
    s0 = _scene(None, _g((1, 1)))
    s1 = _scene(s0, np.zeros((6, 6), dtype=int).tolist())
    assert compute_effect(s0, s1).kind == "disappeared"


def test_structural_effect_when_many_changes():
    g0 = np.zeros((6, 6), dtype=int); g0[0, 0] = 3; g0[5, 5] = 4
    g1 = np.zeros((6, 6), dtype=int); g1[0, 1] = 3; g1[4, 5] = 4
    s0 = _scene(None, g0.tolist())
    s1 = _scene(s0, g1.tolist())
    assert compute_effect(s0, s1).kind == "structural"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_effect.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar `compute_effect`**

```python
# agents/causal/causal_model.py
from __future__ import annotations

from collections import namedtuple

Effect = namedtuple("Effect", "kind detail")


def compute_effect(prev, curr) -> Effect:
    prev_by_id = {o.id: o for o in prev.objects}
    curr_by_id = {o.id: o for o in curr.objects}
    changes = []
    for oid, po in prev_by_id.items():
        co = curr_by_id.get(oid)
        if co is None:
            changes.append(Effect("disappeared", oid))
        else:
            dr = round(co.centroid[0] - po.centroid[0])
            dc = round(co.centroid[1] - po.centroid[1])
            if (dr, dc) != (0, 0):
                changes.append(Effect("moved", (dr, dc)))
            elif co.color != po.color:
                changes.append(Effect("recolored", (po.color, co.color)))
    for oid, co in curr_by_id.items():
        if oid not in prev_by_id:
            changes.append(Effect("appeared", oid))
    if not changes:
        return Effect("none", None)
    if len(changes) == 1:
        return changes[0]
    return Effect("structural", tuple(c.kind for c in changes))
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_effect.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/causal_model.py tests/causal/test_effect.py
git commit -m "feat(causal): compute_effect (diff simbolico de cenas por id)"
```

---

### Task 4: `CausalModel` — observe / predict / stats / serialização

**Files:**
- Modify: `agents/causal/causal_model.py`
- Test: `tests/causal/test_causal_model.py`

**Interfaces:**
- Consumes: `Effect`, `compute_effect` (Task 3).
- Produces: `class CausalModel`:
  - `_bump(action_key:str, effect:Effect, level_up:bool=False)` — incrementa `rules[action_key][effect_token]`; marca progresso.
  - `observe(prev:Scene, action_key:str, curr:Scene, level_up:bool=False) -> Effect` — computa efeito e faz `_bump`.
  - `predict(action_key:str) -> tuple[Effect|None, float]` — efeito modal + confiança (contagem_modal/total); `(None, 0.0)` se inédito.
  - `is_progress(action_key:str) -> bool`.
  - `record_prediction(predicted:Effect|None, actual:Effect)` — atualiza acurácia.
  - `stats() -> dict` — `coverage_keys`, `n_rules`, `stable_ratio` (conf ≥ 0.8), `prediction_accuracy`.
  - `to_dict()/from_dict(d)` — round-trip JSON.
  - Nota: `action_key` é string produzida pela Policy (Task 5), ex. `"ACTION1"` ou `"ACTION6@color=7,size=4"`. Após serialização, `predict().detail` volta como string (só `.kind` é usado a jusante).

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_causal_model.py
import json
from agents.causal.causal_model import CausalModel, Effect


def test_observe_and_predict_modal_effect():
    m = CausalModel()
    m._bump("ACTION1", Effect("moved", (0, 1)))
    m._bump("ACTION1", Effect("moved", (0, 1)))
    m._bump("ACTION1", Effect("none", None))
    eff, conf = m.predict("ACTION1")
    assert eff.kind == "moved"
    assert abs(conf - 2 / 3) < 1e-9


def test_predict_unknown_key():
    assert CausalModel().predict("ACTION9") == (None, 0.0)


def test_progress_flag():
    m = CausalModel()
    m._bump("ACTION2", Effect("structural", ("disappeared",)), level_up=True)
    assert m.is_progress("ACTION2") is True
    assert m.is_progress("ACTION1") is False


def test_prediction_accuracy_tracking():
    m = CausalModel()
    m.record_prediction(Effect("moved", (0, 1)), Effect("moved", (0, 1)))  # acerto
    m.record_prediction(Effect("none", None), Effect("moved", (0, 1)))     # erro
    assert abs(m.stats()["prediction_accuracy"] - 0.5) < 1e-9


def test_serialization_roundtrip():
    m = CausalModel()
    m._bump("ACTION1", Effect("moved", (0, 1)), level_up=True)
    d = json.loads(json.dumps(m.to_dict()))
    m2 = CausalModel.from_dict(d)
    assert m2.predict("ACTION1")[0].kind == "moved"
    assert m2.is_progress("ACTION1") is True
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_causal_model.py -v`
Expected: FAIL (`cannot import name 'CausalModel'`).

- [ ] **Step 3: Implementar `CausalModel`**

Adicionar em `causal_model.py`:

```python
def _effect_token(e: Effect) -> str:
    return f"{e.kind}:{e.detail}"


class CausalModel:
    def __init__(self):
        self.rules = {}                 # action_key -> {effect_token -> count}
        self.progress_keys = set()      # action_keys que levaram a level_up
        self._pred_hits = 0
        self._pred_total = 0

    def _bump(self, action_key, effect: Effect, level_up: bool = False):
        tok = _effect_token(effect)
        d = self.rules.setdefault(action_key, {})
        d[tok] = d.get(tok, 0) + 1
        if level_up:
            self.progress_keys.add(action_key)

    def observe(self, prev, action_key, curr, level_up=False) -> Effect:
        eff = compute_effect(prev, curr)
        self._bump(action_key, eff, level_up)
        return eff

    def predict(self, action_key):
        d = self.rules.get(action_key)
        if not d:
            return (None, 0.0)
        tok, cnt = max(d.items(), key=lambda kv: kv[1])
        total = sum(d.values())
        kind, detail = tok.split(":", 1)
        return (Effect(kind, detail), cnt / total)

    def is_progress(self, action_key) -> bool:
        return action_key in self.progress_keys

    def record_prediction(self, predicted, actual: Effect):
        self._pred_total += 1
        if predicted is not None and predicted.kind == actual.kind:
            self._pred_hits += 1

    def stats(self) -> dict:
        stable = 0
        for d in self.rules.values():
            total = sum(d.values())
            if total and max(d.values()) / total >= 0.8:
                stable += 1
        return {
            "coverage_keys": len(self.rules),
            "n_rules": sum(len(d) for d in self.rules.values()),
            "stable_ratio": stable / len(self.rules) if self.rules else 0.0,
            "prediction_accuracy": self._pred_hits / self._pred_total if self._pred_total else 0.0,
        }

    def to_dict(self) -> dict:
        return {
            "rules": self.rules,
            "progress_keys": sorted(self.progress_keys),
            "pred_hits": self._pred_hits,
            "pred_total": self._pred_total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CausalModel":
        m = cls()
        m.rules = {k: dict(v) for k, v in d.get("rules", {}).items()}
        m.progress_keys = set(d.get("progress_keys", []))
        m._pred_hits = d.get("pred_hits", 0)
        m._pred_total = d.get("pred_total", 0)
        return m
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_causal_model.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/causal_model.py tests/causal/test_causal_model.py
git commit -m "feat(causal): CausalModel observe/predict/stats/serializacao"
```

---

### Task 5: `Policy` — geração de candidatos + chave de ação

**Files:**
- Create: `agents/causal/policy.py`
- Test: `tests/causal/test_policy_candidates.py`

**Interfaces:**
- Consumes: `Scene`/`Object` (perception), `object_at`.
- Produces:
  - `Candidate = namedtuple("Candidate", "action x y key")` (`x,y` = None para ações simples).
  - `def action_key(action, target_obj) -> str` — `action.name` para simples; `f"{action.name}@color={o.color},size={o.size}"` para complexas com objeto (ou `f"{action.name}@empty"`).
  - `def candidates(scene:Scene, available_actions:list) -> list[Candidate]` — 1 candidato por ação simples; para cada ação complexa, um candidato por **centroide de objeto** (int, x=col,y=row). Sem varrer 4096 coords.

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_policy_candidates.py
import numpy as np
from arcengine import GameAction
from agents.causal.perception import parse, match_objects
from agents.causal.policy import candidates


def _scene():
    g = np.zeros((6, 6), dtype=int)
    g[1, 1] = 3
    g[4, 4] = 7
    return match_objects(None, parse(g.tolist()))


def test_simple_action_one_candidate():
    cands = candidates(_scene(), [GameAction.ACTION1])
    assert len(cands) == 1
    assert cands[0].x is None and cands[0].key == "ACTION1"


def test_complex_action_candidate_per_object():
    # ACTION6 é complexa (is_complex() == True); espera 1 candidato por objeto (2)
    cands = candidates(_scene(), [GameAction.ACTION6])
    assert len(cands) == 2
    coords = sorted((c.x, c.y) for c in cands)
    assert coords == [(1, 1), (4, 4)]        # (col,row) dos centroides
    assert all("@color=" in c.key for c in cands)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_policy_candidates.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar geração de candidatos**

```python
# agents/causal/policy.py
from __future__ import annotations

import random
from collections import namedtuple

Candidate = namedtuple("Candidate", "action x y key")


def action_key(action, target_obj) -> str:
    if not action.is_complex():
        return action.name
    if target_obj is None:
        return f"{action.name}@empty"
    return f"{action.name}@color={target_obj.color},size={target_obj.size}"


def candidates(scene, available_actions) -> list:
    out = []
    for action in available_actions:
        if not action.is_complex():
            out.append(Candidate(action, None, None, action.name))
        else:
            for o in scene.objects:
                y = int(round(o.centroid[0]))   # row
                x = int(round(o.centroid[1]))    # col
                out.append(Candidate(action, x, y, action_key(action, o)))
    return out
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_policy_candidates.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/policy.py tests/causal/test_policy_candidates.py
git commit -m "feat(causal): Policy candidatos ancorados em objetos + action_key"
```

---

### Task 6: `Policy` — scoring, FSM EXPLORAR/EXPLOITAR e `decide`

**Files:**
- Modify: `agents/causal/policy.py`
- Test: `tests/causal/test_policy_decide.py`

**Interfaces:**
- Consumes: `candidates`, `Candidate` (Task 5); `CausalModel` (predict/is_progress).
- Produces: `class Policy`:
  - `__init__(self, seed:int=0, epsilon:float=0.05)`.
  - `score(cand:Candidate, model, seen_effects:set, budget_frac:float) -> float` — 4 termos do spec: progresso `+10.0 * (1 + (1 - budget_frac))` se `is_progress`; ganho de informação `+3.0` inédito / `+1.5` conf<0.8; novidade `+0.5` se `eff.kind not in seen_effects`; estagnação `-2.0` se `eff.kind == "none"`.
  - `decide(scene, model, available_actions, seen_effects, budget_frac) -> Candidate` — `argmax(score)` (desempate por ordem de geração); com prob. `epsilon`, candidato aleatório (seed fixa).
  - **Simplificação vs. spec §4.3:** o termo "novidade de estado" do spec (Scene prevista inédita vs. Scenes vistas por hash) é aproximado aqui por **novidade do *tipo de efeito*** (`eff.kind not in seen_effects`) — mais barato e suficiente no v1 (peso baixo). Novidade de Scene por hash fica para refino posterior.

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_policy_decide.py
import numpy as np
from arcengine import GameAction
from agents.causal.perception import parse, match_objects
from agents.causal.causal_model import CausalModel, Effect
from agents.causal.policy import Policy


def _scene():
    g = np.zeros((6, 6), dtype=int); g[1, 1] = 3
    return match_objects(None, parse(g.tolist()))


def test_prefers_untried_over_known_none():
    m = CausalModel()
    m._bump("ACTION1", Effect("none", None))          # conhecido, inútil
    p = Policy(seed=1, epsilon=0.0)
    cand = p.decide(_scene(), m, [GameAction.ACTION1, GameAction.ACTION2], set(), budget_frac=1.0)
    assert cand.action is GameAction.ACTION2          # inédito ganha


def test_prefers_progress_action():
    m = CausalModel()
    m._bump("ACTION1", Effect("structural", ("disappeared",)), level_up=True)  # progresso!
    m._bump("ACTION2", Effect("moved", (0, 1)))
    p = Policy(seed=1, epsilon=0.0)
    cand = p.decide(_scene(), m, [GameAction.ACTION1, GameAction.ACTION2], set(), budget_frac=0.2)
    assert cand.action is GameAction.ACTION1


def test_epsilon_zero_is_deterministic():
    m = CausalModel()
    p = Policy(seed=1, epsilon=0.0)
    args = (_scene(), m, [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3], set(), 1.0)
    assert p.decide(*args).action is p.decide(*args).action
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_policy_decide.py -v`
Expected: FAIL (`cannot import name 'Policy'`).

- [ ] **Step 3: Implementar `Policy.score` e `decide`**

Adicionar em `policy.py`:

```python
class Policy:
    def __init__(self, seed: int = 0, epsilon: float = 0.05):
        self._rng = random.Random(seed)
        self.epsilon = epsilon

    def score(self, cand, model, seen_effects, budget_frac) -> float:
        eff, conf = model.predict(cand.key)
        s = 0.0
        if model.is_progress(cand.key):
            s += 10.0 * (1 + (1 - budget_frac))
        if eff is None:
            s += 3.0
        elif conf < 0.8:
            s += 1.5
        if eff is not None and eff.kind not in seen_effects:
            s += 0.5
        if eff is not None and eff.kind == "none":
            s -= 2.0
        return s

    def decide(self, scene, model, available_actions, seen_effects, budget_frac):
        cands = candidates(scene, available_actions)
        if not cands:
            return None
        if self._rng.random() < self.epsilon:
            return self._rng.choice(cands)
        best, best_s = None, None
        for c in cands:
            sc = self.score(c, model, seen_effects, budget_frac)
            if best_s is None or sc > best_s:
                best, best_s = c, sc
        return best
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_policy_decide.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/policy.py tests/causal/test_policy_decide.py
git commit -m "feat(causal): Policy score de 4 termos + decide (explore/exploit por budget)"
```

---

### Task 7: `Instrumentation` — logger JSONL + métricas

**Files:**
- Create: `agents/causal/instrumentation.py`
- Test: `tests/causal/test_instrumentation.py`

**Interfaces:**
- Consumes: `Effect` (só lê `.kind`).
- Produces: `class Instrumentation`:
  - `__init__(self, path:str|None=None)` — sem `path`, só acumula em `self.records`.
  - `log(self, action_name:str, x, y, mode:str, predicted, actual, model_stats:dict, reasoning:dict) -> None` — anexa dict e, se `path`, escreve linha JSON (append).
  - `summary(self) -> dict` — `n_actions`, `explore_vs_exploit` (contagens por `mode`), `wasted` (nº `actual.kind == "none"`), `last_model_stats`.

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_instrumentation.py
import json
from agents.causal.causal_model import Effect
from agents.causal.instrumentation import Instrumentation


def test_logs_accumulate_and_summarize():
    ins = Instrumentation()
    ins.log("ACTION1", None, None, "EXPLORE",
            None, Effect("moved", (0, 1)), {"coverage_keys": 1}, {"why": "novo"})
    ins.log("ACTION1", None, None, "EXPLOIT",
            Effect("none", None), Effect("none", None), {"coverage_keys": 1}, {"why": "x"})
    s = ins.summary()
    assert s["n_actions"] == 2
    assert s["explore_vs_exploit"] == {"EXPLORE": 1, "EXPLOIT": 1}
    assert s["wasted"] == 1


def test_writes_jsonl(tmp_path):
    p = tmp_path / "log.jsonl"
    ins = Instrumentation(str(p))
    ins.log("ACTION2", 3, 4, "EXPLORE", None, Effect("appeared", 5), {}, {})
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["action"] == "ACTION2" and rec["x"] == 3 and rec["actual"] == "appeared"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_instrumentation.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar `Instrumentation`**

```python
# agents/causal/instrumentation.py
from __future__ import annotations

import json
from collections import Counter


def _kind(effect):
    return None if effect is None else effect.kind


class Instrumentation:
    def __init__(self, path: str | None = None):
        self.path = path
        self.records = []

    def log(self, action_name, x, y, mode, predicted, actual, model_stats, reasoning):
        rec = {
            "action": action_name,
            "x": x,
            "y": y,
            "mode": mode,
            "predicted": _kind(predicted),
            "actual": _kind(actual),
            "model_stats": model_stats,
            "reasoning": reasoning,
        }
        self.records.append(rec)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def summary(self) -> dict:
        modes = Counter(r["mode"] for r in self.records)
        wasted = sum(1 for r in self.records if r["actual"] == "none")
        return {
            "n_actions": len(self.records),
            "explore_vs_exploit": dict(modes),
            "wasted": wasted,
            "last_model_stats": self.records[-1]["model_stats"] if self.records else {},
        }
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_instrumentation.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/instrumentation.py tests/causal/test_instrumentation.py
git commit -m "feat(causal): Instrumentation JSONL + summary (explore/exploit, wasted)"
```

---

### Task 8: `CausalObjectAgent` — orquestração completa no `choose_action`

**Files:**
- Modify: `agents/causal/agent.py`
- Test: `tests/causal/test_agent_integration.py`

**Interfaces:**
- Consumes: `parse`, `match_objects` (perception); `CausalModel` (observe/predict/record_prediction/stats/is_progress); `Policy` (decide); `Instrumentation` (log).
- Produces: `CausalObjectAgent.choose_action` que percebe a cena, casa identidades, **observa a transição da ação anterior** (fecha o loop causal), decide a próxima ação, anexa `reasoning`, loga, e guarda estado (`_prev_scene`, `_last_key`, `_last_predicted`, `_last_level`). `is_done` inalterado. `_init_causal_state` agora cria `_model`, `_policy`, `_instr`, `_seen_effects`.

- [ ] **Step 1: Escrever o teste de integração (falha)**

```python
# tests/causal/test_agent_integration.py
import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.causal.agent import CausalObjectAgent


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.frames = []
    a.action_counter = 0
    a._init_causal_state()
    return a


def _frame(grid, level=0, actions=(GameAction.ACTION1, GameAction.ACTION2)):
    return FrameData(
        levels_completed=level,
        state=GameState.NOT_FINISHED,
        frame=[grid],
        available_actions=list(actions),
    )


def test_closes_causal_loop_across_two_steps():
    a = _agent()
    g0 = np.zeros((6, 6), dtype=int); g0[1, 1] = 3
    f0 = _frame(g0.tolist())
    act0 = a.choose_action([f0], f0)          # 1o passo: sem transição ainda
    assert act0 in f0.available_actions
    g1 = np.zeros((6, 6), dtype=int); g1[1, 2] = 3   # objeto moveu
    f1 = _frame(g1.tolist())
    a.choose_action([f0, f1], f1)             # observa a transição de act0
    assert a._model.stats()["coverage_keys"] >= 1


def test_detects_level_up_as_progress():
    a = _agent()
    g0 = np.zeros((6, 6), dtype=int); g0[1, 1] = 3
    f0 = _frame(g0.tolist(), level=0)
    a.choose_action([f0], f0)
    g1 = np.zeros((6, 6), dtype=int); g1[5, 5] = 4
    f1 = _frame(g1.tolist(), level=1)         # levels_completed subiu
    a.choose_action([f0, f1], f1)
    assert len(a._model.progress_keys) >= 1
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/test_agent_integration.py -v`
Expected: FAIL (AttributeError: `_model`).

- [ ] **Step 3: Implementar a orquestração**

Substituir os imports do topo, `_init_causal_state` e `choose_action` em `agent.py`:

```python
# topo de agents/causal/agent.py — adicionar aos imports existentes:
from .perception import parse, match_objects
from .causal_model import CausalModel
from .policy import Policy
from .instrumentation import Instrumentation
```

```python
def _init_causal_state(self) -> None:
    self._model = CausalModel()
    self._policy = Policy(seed=0, epsilon=0.05)
    self._instr = Instrumentation()
    self._prev_scene = None
    self._last_key = None
    self._last_predicted = None
    self._last_level = 0
    self._seen_effects = set()


def choose_action(self, frames, latest_frame):
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
```

- [ ] **Step 4: Rodar toda a suíte**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/ -v`
Expected: PASS (todos os testes das Tasks 0–8).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_integration.py
git commit -m "feat(causal): CausalObjectAgent orquestra percepcao->modelo->policy (loop causal)"
```

---

### Task 9: Integração local + extensão do `analysis/replay.py`

**Files:**
- Modify: `analysis/replay.py` (plot das curvas de acurácia/cobertura a partir do JSONL — só se o arquivo existir)
- Create: `docs/superpowers/notes/2026-08-27-causal-v1-baseline.md`

**Interfaces:**
- Consumes: JSONL do `Instrumentation` (campos `action, x, y, mode, predicted, actual, model_stats, reasoning`).
- Produces: gráfico `analysis/out/causal_metrics.png` (acurácia + cobertura por passo) e uma nota com os números do baseline vs. random.

- [ ] **Step 1: Rodar a suíte de unit tests completa**

Run: `cd ARC-AGI-3-Agents && uv run pytest tests/causal/ -v`
Expected: PASS (todos).

- [ ] **Step 2: Integração local contra a API (requer ARC_API_KEY no `.env`)**

Run: `cd ARC-AGI-3-Agents && uv run main.py --agent=causalobject --game=ls20`
Expected: executa até `WIN` ou esgotar `MAX_ACTIONS`; gera recording. Se faltar `ARC_API_KEY`, pular e anotar na nota.

Nota: para produzir o JSONL, ligar o `Instrumentation` com `path` (ex.: variável de ambiente `CAUSAL_LOG=analysis/out/causal.jsonl` lida em `_init_causal_state`). Fold desta fiação nesta task se ainda não existir.

- [ ] **Step 3: Estender `analysis/replay.py`**

Adicionar função que plota `prediction_accuracy` e `coverage_keys` por passo a partir do JSONL, integrada como flag opcional `--causal-log <path>` no argparse existente do `replay.py` (seguir o padrão do arquivo):

```python
def plot_causal_metrics(jsonl_path, out_path):
    import json
    import matplotlib.pyplot as plt
    acc, cov = [], []
    with open(jsonl_path) as f:
        for line in f:
            ms = json.loads(line).get("model_stats", {})
            acc.append(ms.get("prediction_accuracy", 0.0))
            cov.append(ms.get("coverage_keys", 0))
    fig, ax1 = plt.subplots()
    ax1.plot(acc, label="prediction_accuracy")
    ax1.set_ylabel("accuracy"); ax1.set_xlabel("step")
    ax2 = ax1.twinx(); ax2.plot(cov, "--", color="tab:orange", label="coverage_keys")
    ax2.set_ylabel("coverage (keys)")
    fig.legend(loc="upper left"); fig.savefig(out_path, dpi=120)
```

- [ ] **Step 4: Registrar o baseline na nota**

Preencher `docs/superpowers/notes/2026-08-27-causal-v1-baseline.md` com: níveis completados por jogo, ações-por-nível, acurácia de previsão final, cobertura — comparado ao agente `random`. Aplicar o **gate de "v1 pronto"** (spec §5): ≥1 nível onde o random falha **E** acurácia >70% em jogos com ≥20 passos.

- [ ] **Step 5: Commit**

```bash
git add analysis/replay.py docs/superpowers/notes/2026-08-27-causal-v1-baseline.md
git commit -m "feat(causal): metricas de modelo no replay.py + nota de baseline v1"
```

---

## Notas de execução

- **Ordem:** as tasks são sequenciais (cada uma consome interfaces da anterior). Perception (1–2), model (3–4), policy (5–6) e instrumentation (7) são internamente coesas; Task 8 integra tudo.
- **Sinal de "objeto" pode divergir do jogo:** se a integração local mostrar segmentação ruim (spec §8), iterar em `Perception` antes de investir na Policy.
- **`frame` em pilha:** `parse` já usa a última camada; validar com `replay.py` se surgir comportamento estranho no fim de nível (bug "winframe" do duck).
- **Objetos multicoloridos:** o v1 segmenta por cor (cada cor = um objeto). Se um "objeto do jogo" for multicor, ele aparecerá como vários — aceitável no v1; refino fica para depois.
