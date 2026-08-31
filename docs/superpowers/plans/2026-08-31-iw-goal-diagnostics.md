# IW Goal-Directed Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose 5 diagnostic keys in `phase2_stats` so an offline Kaggle run reveals *why* the goal-directed IW never completes a level (predicate learned? IW reaches goal? predicate fires on real states?).

**Architecture:** Pure telemetry, additive, no behavior change. Four integer counters + one string field on `CausalObjectAgent`, incremented in `_iw_decide` (IW goal hits/calls) and a new `_eval_reward_real` helper called once per `choose_action` (real-scene reward evaluation), all surfaced via `phase2_stats`.

**Tech Stack:** Python 3.12, pytest. Everything in `agents/causal/agent.py` + tests. No new modules, no changes to `iw.py`/`goals.py`.

## Global Constraints

- Additive telemetry only — no decision/behavior change; the existing decision stack, IW, synthesis, and prompts stay byte-identical.
- Counters move only under `CAUSAL_IW` on + reward learned — same gating as today.
- Reward evaluation on real scenes MUST be exception-safe (reuse `goal_fn_from_reward`, which swallows exceptions from LLM-authored code).
- Keep the full suite green (baseline 312 tests on `main` `6f21256`).
- Do NOT hardcode `ARC_API_KEY` or any secret anywhere.
- `self._reward_src` already inits to `None` at `agent.py:95` and is assigned in `_try_learn_reward` at `agent.py:370` — do not duplicate the init; add only the four new counters beside it.

---

### Task 1: IW goal-directed counting + phase2_stats keys

**Files:**
- Modify: `agents/causal/agent.py` — `_init_causal_state` (add 4 counters after line 95), `_iw_decide` (lines 302-310), `phase2_stats` (lines 374-383)
- Test: `tests/causal/test_agent_iw_diag.py` (create)

**Interfaces:**
- Consumes: `goal_fn_from_reward` (already imported at `agent.py:24`), `iw_plan` (imported at `agent.py:23`), `self._typed.sources` (dict, non-empty ⇒ IW runs), `self._reward_fn` (callable | None).
- Produces: instance attrs `self._iw_goal_calls: int`, `self._iw_goal_hits: int`, `self._reward_real_true: int`, `self._reward_real_evals: int`; `phase2_stats()` dict gains keys `reward_src`, `iw_goal_calls`, `iw_goal_hits`, `reward_real_true`, `reward_real_evals`. The `reward_real_*` counters init here (stay 0) and are incremented in Task 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/causal/test_agent_iw_diag.py`:

```python
import json

import numpy as np

from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects
from agents.causal.policy import Candidate


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def _grid_at(col):
    g = np.zeros((8, 8), dtype=int)
    g[1, col] = 3
    return g


def _scene():
    return match_objects(None, parse(_grid_at(1)))


def _cands():
    return [Candidate(None, None, None, "ACTION1", False)]


# --- _iw_decide conta call+hit quando o IW acha caminho até a meta ---
def test_iw_decide_counts_goal_hit(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._typed.sources = {"shp": "def transition(obj, action, ctx): return obj"}
    a._reward_fn = lambda state: (0.0, True)
    monkeypatch.setattr("agents.causal.agent.iw_plan",
                        lambda *args, **kw: "ACTION1")
    out = a._iw_decide(_scene(), _cands())
    assert out == "ACTION1"
    assert a._iw_goal_calls == 1
    assert a._iw_goal_hits == 1


# --- _iw_decide conta call mas NÃO hit quando o IW não acha caminho (None) ---
def test_iw_decide_counts_goal_miss(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._typed.sources = {"shp": "def transition(obj, action, ctx): return obj"}
    a._reward_fn = lambda state: (0.0, False)
    monkeypatch.setattr("agents.causal.agent.iw_plan",
                        lambda *args, **kw: None)
    out = a._iw_decide(_scene(), _cands())
    assert out is None
    assert a._iw_goal_calls == 1
    assert a._iw_goal_hits == 0


# --- sem regras aceitas: retorna None cedo, gf nunca montado, contadores não mexem ---
def test_iw_decide_no_rules_no_count(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._reward_fn = lambda state: (0.0, True)   # há reward, mas não há f_τ
    out = a._iw_decide(_scene(), _cands())
    assert out is None
    assert a._iw_goal_calls == 0
    assert a._iw_goal_hits == 0


# --- phase2_stats expõe as 5 chaves novas + reward_src ---
def test_phase2_stats_has_diag_keys(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_src = "def reward_function(state): return (0, False)"
    a._iw_goal_calls = 4
    a._iw_goal_hits = 1
    a._reward_real_true = 2
    a._reward_real_evals = 9
    s = a.phase2_stats()
    assert s["reward_src"] == "def reward_function(state): return (0, False)"
    assert s["iw_goal_calls"] == 4
    assert s["iw_goal_hits"] == 1
    assert s["reward_real_true"] == 2
    assert s["reward_real_evals"] == 9
    # continua serializável em JSON (grava em causal_phase2.json)
    json.dumps(s)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-diagnostics && uv run pytest tests/causal/test_agent_iw_diag.py -v`
Expected: FAIL — `AttributeError: 'CausalObjectAgent' object has no attribute '_iw_goal_calls'` (init missing) and `KeyError`/assert on the new phase2_stats keys.

- [ ] **Step 3: Add the four counters in `_init_causal_state`**

In `agents/causal/agent.py`, after line 95 (`self._reward_src = None`), add:

```python
        self._iw_goal_calls = 0       # diag: IW rodou goal-directed (reward viva)
        self._iw_goal_hits = 0        # diag: IW achou caminho até a meta
        self._reward_real_true = 0    # diag: reward_fn deu goal_flag=True em cena real
        self._reward_real_evals = 0   # diag: cenas reais avaliadas pela reward_fn
```

- [ ] **Step 4: Add counting in `_iw_decide`**

Replace the last line of `_iw_decide` (currently `agent.py:310`):

```python
        return iw_plan(start, [c.key for c in cands], self._typed, goal_fn=gf, max_nodes=300)
```

with:

```python
        r = iw_plan(start, [c.key for c in cands], self._typed, goal_fn=gf, max_nodes=300)
        if gf is not None:                       # diag: IW goal-directed disparou
            self._iw_goal_calls += 1
            if r is not None:                    # achou caminho até a meta
                self._iw_goal_hits += 1
        return r
```

- [ ] **Step 5: Add the 5 keys to `phase2_stats`**

In `phase2_stats` (`agent.py:374-383`), after the `"reward_learned": ...` line and before `"eta_rows": ...`, insert:

```python
            "reward_src": self._reward_src,
            "iw_goal_calls": self._iw_goal_calls,
            "iw_goal_hits": self._iw_goal_hits,
            "reward_real_true": self._reward_real_true,
            "reward_real_evals": self._reward_real_evals,
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-diagnostics && uv run pytest tests/causal/test_agent_iw_diag.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Run the full suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-diagnostics && uv run pytest tests/causal tests/kaggle -q`
Expected: PASS, count = 312 baseline + 4 new = 316.

- [ ] **Step 8: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_iw_diag.py
git commit -m "feat: IW goal-directed telemetry (iw_goal_calls/hits + reward_src) em phase2_stats"
```

---

### Task 2: Real-scene reward evaluation

**Files:**
- Modify: `agents/causal/agent.py` — add `_eval_reward_real` method (near `_iw_decide`, ~line 311) and call it in `choose_action` right after `self._percept.observe(...)` (line 135)
- Test: `tests/causal/test_agent_iw_diag.py` (append)

**Interfaces:**
- Consumes: `self._reward_fn` (callable | None), `goal_fn_from_reward` (imported at `agent.py:24`), `_obj_state` (module-level helper at `agent.py:34`), `self._reward_real_true`/`self._reward_real_evals` (init in Task 1).
- Produces: method `self._eval_reward_real(scene) -> None` that bumps `_reward_real_evals` (always, when reward learned) and `_reward_real_true` (when the learned predicate returns `goal_flag=True` on the real state).

- [ ] **Step 1: Write the failing tests**

Append to `tests/causal/test_agent_iw_diag.py`:

```python
# --- _eval_reward_real conta eval+true quando o predicado dá goal_flag=True ---
def test_eval_reward_real_true(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_fn = lambda state: (1.0, True)
    a._eval_reward_real(_scene())
    assert a._reward_real_evals == 1
    assert a._reward_real_true == 1


# --- goal_flag=False: conta eval mas não true ---
def test_eval_reward_real_false(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_fn = lambda state: (0.0, False)
    a._eval_reward_real(_scene())
    assert a._reward_real_evals == 1
    assert a._reward_real_true == 0


# --- sem reward_fn: não avalia nada ---
def test_eval_reward_real_none(monkeypatch):
    a = _agent(monkeypatch)
    assert a._reward_fn is None
    a._eval_reward_real(_scene())
    assert a._reward_real_evals == 0
    assert a._reward_real_true == 0


# --- predicado que quebra não derruba nem conta true (exception-safe) ---
def test_eval_reward_real_exception_safe(monkeypatch):
    a = _agent(monkeypatch)
    def _boom(state):
        raise ValueError("boom")
    a._reward_fn = _boom
    a._eval_reward_real(_scene())        # não levanta
    assert a._reward_real_evals == 1
    assert a._reward_real_true == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-diagnostics && uv run pytest tests/causal/test_agent_iw_diag.py -k eval_reward_real -v`
Expected: FAIL — `AttributeError: 'CausalObjectAgent' object has no attribute '_eval_reward_real'`.

- [ ] **Step 3: Add the `_eval_reward_real` method**

In `agents/causal/agent.py`, immediately after the `_iw_decide` method (after its `return r`, ~line 315), add:

```python
    def _eval_reward_real(self, scene):
        """Diag (b)vs(c): avalia a reward_fn aprendida na cena REAL observada.
        Reusa goal_fn_from_reward (à prova de exceção). Não altera decisão."""
        if self._reward_fn is None:
            return
        state = [(o.shape_hash, _obj_state(o)) for o in scene.objects]
        self._reward_real_evals += 1
        if goal_fn_from_reward(self._reward_fn)(state):
            self._reward_real_true += 1
```

- [ ] **Step 4: Call it once per playing step in `choose_action`**

In `choose_action`, right after line 135 (`self._percept.observe(len(scene.objects))   # §2: ...`), add:

```python
        self._eval_reward_real(scene)               # diag: reward_fn na cena real
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-diagnostics && uv run pytest tests/causal/test_agent_iw_diag.py -v`
Expected: PASS (8 tests total in the file).

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-diagnostics && uv run pytest tests/causal tests/kaggle -q`
Expected: PASS, count = 312 baseline + 8 new = 320.

- [ ] **Step 7: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_iw_diag.py
git commit -m "feat: reward_fn na cena real (reward_real_true/evals) — isola predicado-errado de meta-inalcançável"
```

---

## Notes for the offline notebook

No change needed to `kaggle/build_offline_notebook.py`: it embeds `agents/causal/agent.py` verbatim, so the new keys flow into the log line `[causal] phase2 stats: {...}` and into `/kaggle/working/causal_phase2.json` automatically. The user regenerates with `uv run python kaggle/build_offline_notebook.py` and reads the 5 keys after the offline run.
