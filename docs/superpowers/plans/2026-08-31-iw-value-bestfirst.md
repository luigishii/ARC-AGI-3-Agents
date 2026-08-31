# IW Best-First sobre Reward Denso Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer o IW planejar rumo ao estado de MAIOR reward escalar (best-first) em vez de exigir `goal_flag=True` binário (inalcançável), usando o reward denso que o LLM já produz.

**Architecture:** Novo adaptador `value_fn_from_reward` em `goals.py`; parâmetro opcional `value_fn` em `iw_search`/`iw_plan` (`iw.py`) que roda um best-first por valor com a mesma poda por novidade do IW; `_iw_decide` (`agent.py`) passa `value_fn`. Precedência `goal_fn` > `value_fn` > exploração.

**Tech Stack:** Python 3.12, pytest. Sem novos módulos, sem deps.

## Global Constraints

- Additive/comportamento-preservador nos outros modos: `goal_fn` de `iw.py` permanece; sem reward, `value_fn=None` → IW volta ao modo exploração width-based de hoje.
- Só muda o caminho já sob `CAUSAL_IW` on + reward aprendida.
- Exception-safe: `value_fn_from_reward` engole exceções do código LLM-autorado (→ `float("-inf")`).
- Manter a suíte verde (baseline 320 na `main` `e373943`).
- Não hardcodar `ARC_API_KEY` nem segredos.
- A propagação da ação-raiz na fila do `iw_search` passa a carregar `first` (a ação-raiz) em TODA expansão — corrige atribuição de raiz para profundidade ≥2 nos dois modos; os testes existentes usam ação única (`["A"]`) então `first=="A"` em qualquer profundidade → sem regressão.

---

### Task 1: `value_fn_from_reward` em goals.py

**Files:**
- Modify: `agents/causal/goals.py` (append após `goal_fn_from_reward`, fim do arquivo linha 46)
- Test: `tests/causal/test_goals_value.py` (create)

**Interfaces:**
- Consumes: nada novo (stdlib).
- Produces: `value_fn_from_reward(reward_fn) -> callable` onde a callable é `value(state) -> float` (lê `reward_fn(state)[0]`; exceção → `float("-inf")`).

- [ ] **Step 1: Write the failing test**

Create `tests/causal/test_goals_value.py`:

```python
import math

from agents.causal.goals import value_fn_from_reward


def test_value_reads_scalar_from_tuple():
    vf = value_fn_from_reward(lambda state: (42.0, False))
    assert vf(["anything"]) == 42.0


def test_value_reads_bare_scalar():
    vf = value_fn_from_reward(lambda state: 7)
    assert vf(["anything"]) == 7.0


def test_value_exception_is_neg_inf():
    def boom(state):
        raise ValueError("boom")
    vf = value_fn_from_reward(boom)
    assert vf(["x"]) == float("-inf")
    assert math.isinf(vf(["x"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-value-bestfirst && uv run pytest tests/causal/test_goals_value.py -v`
Expected: FAIL — `ImportError: cannot import name 'value_fn_from_reward'`.

- [ ] **Step 3: Implement `value_fn_from_reward`**

In `agents/causal/goals.py`, append after the `goal_fn_from_reward` function (end of file):

```python


def value_fn_from_reward(reward_fn):
    """Adapta reward_function -> value(state)->float p/ o IW best-first. Lê o reward
    (1º elemento) e é à prova de exceção (falha → -inf, nunca escolhido)."""
    def value(state):
        try:
            r = reward_fn(state)
            if isinstance(r, (tuple, list)) and len(r) >= 1:
                return float(r[0])
            return float(r)
        except Exception:
            return float("-inf")
    return value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-value-bestfirst && uv run pytest tests/causal/test_goals_value.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/goals.py tests/causal/test_goals_value.py
git commit -m "feat: value_fn_from_reward (scalar denso do reward p/ IW best-first)"
```

---

### Task 2: modo `value_fn` em iw.py

**Files:**
- Modify: `agents/causal/iw.py` — `iw_search` (linhas 47-77) e `iw_plan` (linhas 80-86)
- Test: `tests/causal/test_iw_value.py` (create)

**Interfaces:**
- Consumes: `TypedWorldModel` (`.predict(state, action)`), `_register`/`_new_count`/`atoms` internos.
- Produces: `iw_search(start, actions, model, goal_fn=None, value_fn=None, width=1, max_nodes=1000)` e `iw_plan(start, actions, model, goal_fn=None, value_fn=None, max_width=2, max_nodes=1000)`. Com `value_fn` (e sem `goal_fn`): devolve a ação-raiz rumo ao estado de maior valor alcançado, só se estritamente maior que `value_fn(start)`; senão `None`. `goal_fn` e exploração pura inalterados.

- [ ] **Step 1: Write the failing tests**

Create `tests/causal/test_iw_value.py`:

```python
from agents.causal.iw import iw_search, iw_plan
from agents.causal.typed_model import TypedWorldModel


# regra que ramifica na ação: A → x+1, qualquer outra → x-1
_BRANCH = (
    "def transition(obj, action, ctx):\n"
    "    o = dict(obj)\n"
    "    o['x'] = o['x'] + 1 if action == 'A' else o['x'] - 1\n"
    "    return o\n"
)
# regra monotônica: sempre x+1 (qualquer ação)
_UP = "def transition(obj, action, ctx):\n    o = dict(obj)\n    o['x'] = o['x'] + 1\n    return o\n"


def _model(rule):
    m = TypedWorldModel()
    m.set_rule("t", rule)
    return m


def _start(x=0):
    return [("t", {"x": x, "y": 0, "color": 3})]


# --- best-first escolhe a ação que sobe o valor ---
def test_value_picks_improving_action():
    higher_x = lambda st: float(st[0][1]["x"])
    out = iw_search(_start(0), ["A", "B"], _model(_BRANCH),
                    value_fn=higher_x, width=1)
    assert out == "A"


# --- nada melhora o start → None ---
def test_value_none_when_no_improvement():
    lower_x = lambda st: -float(st[0][1]["x"])   # premia x menor; _UP só aumenta x
    out = iw_search(_start(0), ["A"], _model(_UP),
                    value_fn=lower_x, width=1, max_nodes=50)
    assert out is None


# --- goal_fn tem precedência sobre value_fn (comportamento atual inalterado) ---
def test_goal_fn_takes_precedence_over_value():
    goal = lambda st: st[0][1]["x"] >= 2
    ignored_value = lambda st: -999.0
    out = iw_search(_start(0), ["A"], _model(_UP),
                    goal_fn=goal, value_fn=ignored_value, width=1)
    assert out == "A"


# --- exploração pura (sem goal_fn nem value_fn) inalterada ---
def test_pure_exploration_unchanged():
    out = iw_search(_start(0), ["A"], _model(_UP), width=1)
    assert out == "A"


# --- iw_plan repassa value_fn e escala a largura ---
def test_iw_plan_value_best_first():
    higher_x = lambda st: float(st[0][1]["x"])
    out = iw_plan(_start(0), ["A", "B"], _model(_BRANCH),
                  value_fn=higher_x, max_width=2)
    assert out == "A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-value-bestfirst && uv run pytest tests/causal/test_iw_value.py -v`
Expected: FAIL — `TypeError: iw_search() got an unexpected keyword argument 'value_fn'`.

- [ ] **Step 3: Rewrite `iw_search` and `iw_plan`**

In `agents/causal/iw.py`, replace the whole `iw_search` function (linhas 47-77) with:

```python
def iw_search(start, actions, model, goal_fn=None, value_fn=None, width=1, max_nodes=1000):
    """BFS com poda por novidade de largura `width`.
    - goal_fn → 1ª ação do caminho que atinge o goal (senão None).
    - senão value_fn → 1ª ação rumo ao estado de MAIOR valor alcançado, se estritamente
      maior que value_fn(start) (senão None). Best-first mantendo a poda por novidade.
    - senão (exploração) → ação-raiz cujo próximo estado revela mais átomos inéditos
      (None se nenhuma muda nada)."""
    if goal_fn is not None and goal_fn(start):
        return None
    seen = set()
    _register(start, seen, width)

    val_mode = goal_fn is None and value_fn is not None
    best_val = value_fn(start) if val_mode else None
    best_val_action = None

    q = deque()
    best_action, best_new = None, 0
    for a in actions:
        nxt = model.predict(start, a)
        if goal_fn is None and value_fn is None:
            n = _new_count(nxt, seen, width)
            if n > best_new:
                best_new, best_action = n, a
        q.append((nxt, a))

    nodes = 0
    while q and nodes < max_nodes:
        state, first = q.popleft()
        nodes += 1
        if goal_fn is not None:
            if goal_fn(state):
                return first
        elif val_mode:
            v = value_fn(state)
            if v > best_val:                # estritamente melhor que o start/melhor atual
                best_val, best_val_action = v, first
        if not _register(state, seen, width):
            continue                        # não-novo → poda (coração do IW)
        for a in actions:
            q.append((model.predict(state, a), first))   # propaga a AÇÃO-RAIZ

    if goal_fn is not None:
        return None
    if val_mode:
        return best_val_action              # None se nada superou value_fn(start)
    return best_action
```

Then replace `iw_plan` (linhas 80-86) with:

```python
def iw_plan(start, actions, model, goal_fn=None, value_fn=None, max_width=2, max_nodes=1000):
    """IW iterado: tenta largura 1, depois 2, ... até max_width. 1ª que resolve vence."""
    for w in range(1, max_width + 1):
        r = iw_search(start, actions, model, goal_fn, value_fn, w, max_nodes)
        if r is not None:
            return r
    return None
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-value-bestfirst && uv run pytest tests/causal/test_iw_value.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the existing iw tests to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-value-bestfirst && uv run pytest tests/causal/test_iw.py -v`
Expected: PASS (todos os testes de `test_iw.py` — goal_fn e exploração inalterados).

- [ ] **Step 6: Commit**

```bash
git add agents/causal/iw.py tests/causal/test_iw_value.py
git commit -m "feat: modo value_fn best-first no iw_search/iw_plan (propaga ação-raiz)"
```

---

### Task 3: `_iw_decide` usa `value_fn`

**Files:**
- Modify: `agents/causal/agent.py` — import (linha 24) e `_iw_decide`
- Test: `tests/causal/test_agent_iw_value.py` (create)

**Interfaces:**
- Consumes: `value_fn_from_reward` (Task 1), `iw_plan(value_fn=...)` (Task 2), `self._reward_fn`, `self._typed.sources`, contadores `_iw_goal_calls`/`_iw_goal_hits` (já existem).
- Produces: `_iw_decide` passa `value_fn=vf` (não mais `goal_fn`); `iw_goal_calls` incrementa quando `vf` ativo; `iw_goal_hits` incrementa quando o IW devolve ação (= achou melhoria de valor).

- [ ] **Step 1: Write the failing test**

Create `tests/causal/test_agent_iw_value.py`:

```python
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


def _scene():
    g = np.zeros((8, 8), dtype=int)
    g[1, 1] = 3
    return match_objects(None, parse(g))


def _cands():
    return [Candidate(None, None, None, "ACTION1", False)]


# --- _iw_decide passa value_fn (não goal_fn) e conta call+hit quando há melhoria ---
def test_iw_decide_uses_value_fn(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._typed.sources = {"shp": "def transition(obj, action, ctx): return obj"}
    a._reward_fn = lambda state: (5.0, False)
    captured = {}

    def fake_iw_plan(start, actions, model, goal_fn=None, value_fn=None,
                     max_width=2, max_nodes=1000):
        captured["goal_fn"] = goal_fn
        captured["value_fn"] = value_fn
        return "ACTION1"

    monkeypatch.setattr("agents.causal.agent.iw_plan", fake_iw_plan)
    out = a._iw_decide(_scene(), _cands())
    assert out == "ACTION1"
    assert captured["goal_fn"] is None              # não usa mais goal_fn
    assert callable(captured["value_fn"])           # passa value_fn
    assert a._iw_goal_calls == 1
    assert a._iw_goal_hits == 1


# --- sem melhoria (iw_plan devolve None): call conta, hit não ---
def test_iw_decide_value_miss(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._typed.sources = {"shp": "def transition(obj, action, ctx): return obj"}
    a._reward_fn = lambda state: (5.0, False)
    monkeypatch.setattr("agents.causal.agent.iw_plan", lambda *args, **kw: None)
    out = a._iw_decide(_scene(), _cands())
    assert out is None
    assert a._iw_goal_calls == 1
    assert a._iw_goal_hits == 0


# --- sem regras aceitas: None cedo, contadores não mexem ---
def test_iw_decide_no_rules(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_IW="1")
    a._reward_fn = lambda state: (5.0, False)
    out = a._iw_decide(_scene(), _cands())
    assert out is None
    assert a._iw_goal_calls == 0
    assert a._iw_goal_hits == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-value-bestfirst && uv run pytest tests/causal/test_agent_iw_value.py -v`
Expected: FAIL — `test_iw_decide_uses_value_fn` falha em `captured["value_fn"]` (hoje `_iw_decide` passa `goal_fn`, não `value_fn`).

- [ ] **Step 3: Add the import**

In `agents/causal/agent.py`, line 24, change:

```python
from .goals import compile_reward, static_reward_check, goal_fn_from_reward
```

to:

```python
from .goals import compile_reward, static_reward_check, goal_fn_from_reward, value_fn_from_reward
```

- [ ] **Step 4: Rewrite `_iw_decide` to use `value_fn`**

In `agents/causal/agent.py`, replace the body of `_iw_decide` (the lines from `start = [...]` through `return r`) with:

```python
        start = [(o.shape_hash, _obj_state(o)) for o in scene.objects]
        vf = value_fn_from_reward(self._reward_fn) if self._reward_fn else None
        r = iw_plan(start, [c.key for c in cands], self._typed, value_fn=vf, max_nodes=300)
        if vf is not None:                       # diag: IW value-directed disparou
            self._iw_goal_calls += 1
            if r is not None:                    # achou ação que melhora o valor
                self._iw_goal_hits += 1
        return r
```

Also update the `_iw_decide` docstring (the lines that say "Goal ainda não disponível ... → modo exploração width-based") to reflect value-directed mode:

```python
        """Planeja com Iterated Width sobre o TypedWorldModel (regras f_τ aceitas).
        Sem regras aceitas → None (cai no fallback). Com reward aprendida → best-first
        que sobe o reward denso (value_fn); sem reward → exploração width-based."""
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-value-bestfirst && uv run pytest tests/causal/test_agent_iw_value.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Reconcile the diagnostics test from the previous feature**

`tests/causal/test_agent_iw_diag.py::test_iw_decide_counts_goal_hit` and `::test_iw_decide_counts_goal_miss` monkeypatch `agents.causal.agent.iw_plan` with `lambda *args, **kw: ...` and set `a._reward_fn` — they still pass because the monkeypatch ignores kwargs and `_iw_decide` still calls `iw_plan` + increments the same counters. Run them to confirm:

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-value-bestfirst && uv run pytest tests/causal/test_agent_iw_diag.py -v`
Expected: PASS (8 tests). The monkeypatches use `*args, **kw`, so no change expected.

- [ ] **Step 7: Run the full suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/iw-value-bestfirst && uv run pytest tests/causal tests/kaggle -q`
Expected: PASS, count = 320 baseline + 3 (Task1) + 5 (Task2) + 3 (Task3) = 331.

- [ ] **Step 8: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_iw_value.py
git commit -m "feat: _iw_decide best-first por valor (value_fn) — IW dirigido pelo reward denso"
```

---

## Notes for the offline notebook

`kaggle/build_offline_notebook.py` embute `goals.py`/`iw.py`/`agent.py` verbatim → o novo modo flui pro run offline sem mudança. Após o merge, regenerar com `uv run python kaggle/build_offline_notebook.py`. Validação real = próximo run: observar `iw_goal_hits > 0` e se `levels_completed` sobe.
