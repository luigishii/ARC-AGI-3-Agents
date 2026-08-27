# CausalObjectAgent v10a — LLM Hybrid: contrato · Plano

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Peças puras e mockáveis do contrato LLM: cliente, prompt a partir da percepção, parser de meta JSON, executor de meta → action_key.

**Architecture:** Novo `agents/causal/llm.py` aditivo (não ligado ao agente). Testado com `FakeLLM`. Sem modelo real.

**Tech Stack:** Python 3.12, stdlib puro (`json`, `collections`), pytest.

## Global Constraints

- stdlib puro; sem deps novas; sem GPU/modelo real.
- Tipos de meta: `press`{action}, `click_cell`{gx,gy}, `reach`{avatar,target}; seletor `{id}|{color}|"rarest"`.
- `parse_goal`/`execute_goal` retornam `None` em qualquer desvio (fallback).
- v10a é aditivo: não alterar nenhum módulo existente. 128 testes v1–v8 verdes.

---

### Task 1: `LLMClient` + `build_prompt` + `parse_goal` (`llm.py`)

**Files:**
- Create: `agents/causal/llm.py`
- Test: `tests/causal/test_llm.py` (novo)

**Interfaces:**
- Produces: `LLMClient`, `NullLLMClient`, `build_prompt(scene, dynamics)->str`, `parse_goal(text)->dict|None`, `GOAL_TYPES`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_llm.py
from dataclasses import replace
from agents.causal.perception import Scene, Object
from agents.causal.llm import NullLLMClient, build_prompt, parse_goal


class FakeLLM:
    def __init__(self, canned):
        self.canned = canned
    def complete(self, prompt):
        return self.canned


def _obj(cells, color=3, oid=0):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return replace(Object(color, cset, bbox, centroid, len(cells), "h"), id=oid)


def _scene():
    return Scene(objects=[_obj([(0, 0)], color=3, oid=1),
                          _obj([(0, 5)], color=7, oid=2)], grid=None)


def test_null_and_fake_client():
    assert NullLLMClient().complete("x") == ""
    assert FakeLLM("hi").complete("x") == "hi"


def test_build_prompt_contains_state_and_instruction():
    p = build_prompt(_scene(), {"available": ["ACTION1"], "moves": {"ACTION1": (0, 1)}})
    assert "OBJECTS" in p
    assert "color=3" in p and "color=7" in p
    assert "AVAILABLE_ACTIONS" in p
    assert "moves" in p
    assert "JSON" in p or "type" in p
    assert build_prompt(_scene(), {}) == build_prompt(_scene(), {})   # determinístico


def test_parse_goal_press():
    assert parse_goal('sure: {"type":"press","action":"ACTION1"} done') == \
        {"type": "press", "action": "ACTION1"}


def test_parse_goal_click_and_reach():
    assert parse_goal('{"type":"click_cell","gx":1,"gy":2}') == \
        {"type": "click_cell", "gx": 1, "gy": 2}
    g = parse_goal('{"type":"reach","avatar":{"id":1},"target":"rarest"}')
    assert g["type"] == "reach" and g["target"] == "rarest"


def test_parse_goal_rejects_bad():
    assert parse_goal("no json here") is None
    assert parse_goal("{not valid") is None
    assert parse_goal('{"type":"unknown"}') is None
    assert parse_goal('{"type":"press"}') is None      # falta action
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v10 && uv run pytest tests/causal/test_llm.py -q`
Expected: FAIL (ModuleNotFoundError: `agents.causal.llm`).

- [ ] **Step 3: Write minimal implementation**

```python
# agents/causal/llm.py
from __future__ import annotations

import json
from collections import Counter

GOAL_TYPES = {"press", "click_cell", "reach"}

_INSTRUCTION = (
    "You are playing a grid puzzle. Infer the GOAL and reply with ONLY a JSON "
    "object, one of:\n"
    '{"type":"press","action":"ACTION1"}\n'
    '{"type":"click_cell","gx":0,"gy":0}\n'
    '{"type":"reach","avatar":<sel>,"target":<sel>}  '
    '(sel = {"id":I} | {"color":C} | "rarest")'
)


class LLMClient:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class NullLLMClient(LLMClient):
    def complete(self, prompt: str) -> str:
        return ""


def build_prompt(scene, dynamics) -> str:
    dyn = dynamics or {}
    lines = [f"OBJECTS ({len(scene.objects)}):"]
    for o in scene.objects:
        lines.append(
            f"  id={o.id} color={o.color} centroid={o.centroid} "
            f"size={o.size} bbox={o.bbox}"
        )
    lines.append(f"AVAILABLE_ACTIONS: {dyn.get('available', [])}")
    lines.append(f"DYNAMICS: moves={dyn.get('moves', {})} notes={dyn.get('notes', '')}")
    lines.append(_INSTRUCTION)
    return "\n".join(lines)


def parse_goal(text):
    if not text:
        return None
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        g = json.loads(text[i:j + 1])
    except Exception:
        return None
    if not isinstance(g, dict) or g.get("type") not in GOAL_TYPES:
        return None
    t = g["type"]
    if t == "press" and "action" in g:
        return g
    if t == "click_cell" and "gx" in g and "gy" in g:
        return g
    if t == "reach" and "avatar" in g and "target" in g:
        return g
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v10 && uv run pytest tests/causal/test_llm.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/llm.py tests/causal/test_llm.py
git commit -m "feat(causal): LLM contrato — cliente, build_prompt, parse_goal"
```

---

### Task 2: `execute_goal` + seletores + regressão

**Files:**
- Modify: `agents/causal/llm.py` (`_resolve`, `execute_goal`)
- Test: `tests/causal/test_llm_execute.py` (novo)

**Interfaces:**
- Consumes: `parse_goal` output; cena com `.objects` (`.id/.color/.centroid`).
- Produces: `execute_goal(goal, scene, moves) -> str | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_llm_execute.py
from dataclasses import replace
from agents.causal.perception import Scene, Object
from agents.causal.llm import execute_goal


def _obj(cells, color=3, oid=0):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return replace(Object(color, cset, bbox, centroid, len(cells), "h"), id=oid)


def _scene(objs):
    return Scene(objects=objs, grid=None)


def test_execute_press():
    assert execute_goal({"type": "press", "action": "ACTION1"}, _scene([]), {}) == "ACTION1"


def test_execute_click_cell():
    assert execute_goal({"type": "click_cell", "gx": 1, "gy": 2}, _scene([]), {}) == \
        "ACTION6@cell=1,2"


def test_execute_reach_moves_toward():
    scene = _scene([_obj([(0, 0)], color=3, oid=1), _obj([(0, 5)], color=7, oid=2)])
    goal = {"type": "reach", "avatar": {"id": 1}, "target": {"id": 2}}
    moves = {"ACTION1": (0, 1), "ACTION2": (0, -1)}
    assert execute_goal(goal, scene, moves) == "ACTION1"


def test_execute_reach_rarest_target():
    scene = _scene([
        _obj([(0, 0)], color=5, oid=1),   # avatar
        _obj([(0, 1)], color=3, oid=2),
        _obj([(0, 9)], color=9, oid=3),   # cor rara
        _obj([(9, 9)], color=3, oid=4),
    ])
    goal = {"type": "reach", "avatar": {"id": 1}, "target": "rarest"}
    moves = {"ACTION1": (0, 1), "ACTION2": (0, -1)}
    assert execute_goal(goal, scene, moves) == "ACTION1"


def test_execute_reach_none_without_moves_or_object():
    scene = _scene([_obj([(0, 0)], oid=1)])
    assert execute_goal({"type": "reach", "avatar": {"id": 1}, "target": {"id": 2}},
                        scene, {}) is None                       # sem moves
    assert execute_goal({"type": "reach", "avatar": {"id": 9}, "target": {"id": 1}},
                        scene, {"ACTION1": (0, 1)}) is None       # avatar inexistente
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v10 && uv run pytest tests/causal/test_llm_execute.py -q`
Expected: FAIL (`execute_goal` inexistente).

- [ ] **Step 3: Write minimal implementation**

Adicionar em `agents/causal/llm.py`:

```python
def _resolve(sel, scene):
    if sel == "rarest":
        if not scene.objects:
            return None
        freq = Counter(o.color for o in scene.objects)
        return min(scene.objects, key=lambda o: freq[o.color])
    if isinstance(sel, dict):
        if "id" in sel:
            for o in scene.objects:
                if o.id == sel["id"]:
                    return o
            return None
        if "color" in sel:
            ms = [o for o in scene.objects if o.color == sel["color"]]
            return ms[0] if ms else None
    return None


def execute_goal(goal, scene, moves):
    t = goal.get("type")
    if t == "press":
        return goal.get("action")
    if t == "click_cell":
        return f"ACTION6@cell={goal['gx']},{goal['gy']}"
    if t == "reach":
        if not moves:
            return None
        avatar = _resolve(goal.get("avatar"), scene)
        target = _resolve(goal.get("target"), scene)
        if avatar is None or target is None:
            return None
        ay, ax = avatar.centroid
        ty, tx = target.centroid
        best, bd = None, abs(ty - ay) + abs(tx - ax)
        for k, (dr, dc) in moves.items():
            nd = abs(ty - (ay + dr)) + abs(tx - (ax + dc))
            if nd < bd:
                bd, best = nd, k
        return best
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v10 && uv run pytest tests/causal/test_llm_execute.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Rodar a suíte inteira (regressão)**

Run: `cd .claude/worktrees/causal-v10 && uv run pytest tests/causal tests/kaggle -q`
Expected: PASS (128 v1–v8 + novos v10a). (`tests/unit/` já falha na base — não rodar.)

- [ ] **Step 6: Commit**

```bash
git add agents/causal/llm.py tests/causal/test_llm_execute.py
git commit -m "feat(causal): execute_goal (meta declarativa → action_key) + regressão verde"
```

---

## Fora de escopo (v10b/c/d)

- Serving real (Qwen2.5-Coder-7B), notebook, pesos como Dataset (v10b).
- Sandbox de código (v10c); controlador + wiring no agente (v10d).
