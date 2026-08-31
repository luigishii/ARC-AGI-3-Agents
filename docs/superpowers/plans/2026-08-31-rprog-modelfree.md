# Progresso Model-Free por Reward Real (Lever B′) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Escolher a ação pelo delta de reward REAL observado (não pela simulação do world-model, que é cego à variável do reward), fazendo hill-climbing model-free sobre o reward denso do LLM.

**Architecture:** Tracker no close-loop acumula `Δ = value(depois) − value(antes)` por action_key nas transições decisão→decisão; camada `_rprog_decide` na pilha de decisão escolhe a key de maior Δ médio positivo; diagnóstico em `phase2_stats`; toggle `CAUSAL_RPROG` nos 2 builders de notebook.

**Tech Stack:** Python 3.12, pytest. Só `agents/causal/agent.py` + `kaggle/build_notebook.py` + `kaggle/build_offline_notebook.py` + testes. Sem novos módulos.

## Global Constraints

- Default-safe: tracker só roda com reward aprendida; camada só sob `CAUSAL_RPROG`. Desligado → pilha idêntica de hoje.
- Reuso: `value_fn_from_reward` já existe em `goals.py` e já está importado em `agent.py:24`.
- À prova de valor inválido: descartar amostra com valor não-finito (`math.isfinite`).
- Só medir em transição decisão→decisão (o `else` que já protege `f_τ`/η das transições fabricadas de level-up — `agent.py:156-160`).
- Manter a suíte verde (baseline 331 na `main` `c65ddf8`).
- Não hardcodar `ARC_API_KEY` nem segredos.

---

### Task 1: estado + tracker no close-loop + chaves de diagnóstico

**Files:**
- Modify: `agents/causal/agent.py` — `import math` (topo); `_init_causal_state` (após os contadores de diag, ~linha 99); close-loop (após `self._observe_types(...)`, `agent.py:159`); `phase2_stats`
- Test: `tests/causal/test_agent_rprog.py` (create)

**Interfaces:**
- Consumes: `value_fn_from_reward` (import em `agent.py:24`), `_obj_state` (`agent.py:34`), `self._reward_fn`, `self._prev_scene`, `self._last_key`.
- Produces: `self._rprog: dict[str, list]` (`key → [soma_Δ, contagem]`), `self._rprog_fires: int`, `self._rprog_on: bool`; `phase2_stats` ganha `rprog_actions` e `rprog_fires`. Método de close-loop `_track_rprog(scene)` chamado na transição decisão→decisão.

- [ ] **Step 1: Write the failing tests**

Create `tests/causal/test_agent_rprog.py`:

```python
import numpy as np

from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def _scene(n):
    """Cena com n objetos (n pixels isolados)."""
    g = np.zeros((8, 8), dtype=int)
    for i in range(n):
        g[0, i] = 3
    return match_objects(None, parse(g))


# --- tracker acumula Δ>0 quando o reward sobe (menos objetos) ---
def test_track_rprog_positive_delta(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_fn = lambda state: (100.0 - (len(state) - 1) * 10.0, False)
    a._prev_scene = _scene(3)          # antes: 3 objetos → value 80
    a._last_key = "ACTION6"
    a._track_rprog(_scene(1))          # depois: 1 objeto → value 100 → Δ=+20
    assert a._rprog["ACTION6"][1] == 1
    assert a._rprog["ACTION6"][0] == 20.0


# --- tracker acumula Δ<0 quando o reward cai (mais objetos) ---
def test_track_rprog_negative_delta(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_fn = lambda state: (100.0 - (len(state) - 1) * 10.0, False)
    a._prev_scene = _scene(1)
    a._last_key = "ACTION1"
    a._track_rprog(_scene(3))          # 100 → 80 → Δ=-20
    assert a._rprog["ACTION1"][0] == -20.0


# --- sem reward_fn: não rastreia ---
def test_track_rprog_no_reward(monkeypatch):
    a = _agent(monkeypatch)
    assert a._reward_fn is None
    a._prev_scene = _scene(2)
    a._last_key = "ACTION1"
    a._track_rprog(_scene(1))
    assert a._rprog == {}


# --- valor não-finito é descartado (não polui a média) ---
def test_track_rprog_discards_non_finite(monkeypatch):
    a = _agent(monkeypatch)
    def boom(state):
        raise ValueError("boom")       # value_fn_from_reward → -inf
    a._reward_fn = boom
    a._prev_scene = _scene(2)
    a._last_key = "ACTION1"
    a._track_rprog(_scene(1))
    assert a._rprog == {}


# --- phase2_stats expõe rprog_actions e rprog_fires ---
def test_phase2_stats_rprog_keys(monkeypatch):
    a = _agent(monkeypatch)
    a._rprog = {"ACTION6": [30.0, 3], "ACTION1": [-5.0, 2]}
    a._rprog_fires = 7
    s = a.phase2_stats()
    assert s["rprog_actions"] == 1     # só ACTION6 tem média > 0
    assert s["rprog_fires"] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/rprog-modelfree && uv run pytest tests/causal/test_agent_rprog.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_track_rprog'` / `'_rprog'` e `KeyError: 'rprog_actions'`.

- [ ] **Step 3: Add `import math`**

In `agents/causal/agent.py`, after `import os` (line 1):

```python
import os
import math
```

- [ ] **Step 4: Init the rprog state**

In `_init_causal_state`, after `self._reward_real_evals = 0` (the last diag counter), add:

```python
        self._rprog = {}              # action_key -> [soma_Δ, contagem] (progresso model-free)
        self._rprog_fires = 0         # diag: vezes que a camada rprog escolheu a ação
        self._rprog_on = os.environ.get("CAUSAL_RPROG", "0") != "0"
```

- [ ] **Step 5: Add the `_track_rprog` method**

In `agents/causal/agent.py`, add a method right after `_eval_reward_real`:

```python
    def _track_rprog(self, scene):
        """B′: acumula Δ do reward REAL (depois-antes) por action_key. Model-free —
        contorna o f_τ que não simula a variável do reward. Só decisão→decisão."""
        if self._reward_fn is None or self._last_key is None or self._prev_scene is None:
            return
        vf = value_fn_from_reward(self._reward_fn)
        before = [(o.shape_hash, _obj_state(o)) for o in self._prev_scene.objects]
        after = [(o.shape_hash, _obj_state(o)) for o in scene.objects]
        vb, va = vf(before), vf(after)
        if not (math.isfinite(vb) and math.isfinite(va)):
            return
        row = self._rprog.setdefault(self._last_key, [0.0, 0])
        row[0] += va - vb
        row[1] += 1
```

- [ ] **Step 6: Call it in the decisão→decisão block**

In `choose_action`, right after `self._observe_types(self._prev_scene, scene, self._last_key)` (`agent.py:159`), add (same indentation, inside the `else` block):

```python
                self._track_rprog(scene)          # B′: delta de reward real por ação
```

- [ ] **Step 7: Add the phase2_stats keys**

In `phase2_stats`, after the `"reward_real_evals": ...` line and before `"eta_rows": ...`, insert:

```python
            "rprog_actions": sum(1 for r in self._rprog.values() if r[1] and r[0] / r[1] > 0),
            "rprog_fires": self._rprog_fires,
```

- [ ] **Step 8: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/rprog-modelfree && uv run pytest tests/causal/test_agent_rprog.py -v`
Expected: PASS (5 tests).

- [ ] **Step 9: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_rprog.py
git commit -m "feat: tracker de progresso model-free (Δ reward real por ação) + diag em phase2_stats"
```

---

### Task 2: camada `_rprog_decide` na pilha de decisão

**Files:**
- Modify: `agents/causal/agent.py` — método `_rprog_decide` (near `_track_rprog`) e wiring no `choose_action` (entre navigate e IW, `agent.py:226-233`)
- Test: `tests/causal/test_agent_rprog.py` (append)

**Interfaces:**
- Consumes: `self._rprog`, `self._rprog_on`, `self._rprog_fires`, `cands` (lista de `Candidate` com `.key`).
- Produces: `_rprog_decide(cands) -> key | None` (maior Δ médio positivo; incrementa `rprog_fires`); disparado na pilha sob `CAUSAL_RPROG`, posição `navigate → rprog → IW → plan → η → greedy`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/causal/test_agent_rprog.py`:

```python
from agents.causal.policy import Candidate


def _cands(*keys):
    return [Candidate(None, None, None, k, False) for k in keys]


# --- escolhe a key de maior Δ médio positivo e incrementa rprog_fires ---
def test_rprog_decide_picks_best_positive(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_RPROG="1")
    a._rprog = {"ACTION6": [30.0, 3], "ACTION1": [4.0, 2]}   # médias 10 vs 2
    out = a._rprog_decide(_cands("ACTION1", "ACTION6"))
    assert out == "ACTION6"
    assert a._rprog_fires == 1


# --- nenhuma média positiva → None, rprog_fires inalterado ---
def test_rprog_decide_none_when_no_positive(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_RPROG="1")
    a._rprog = {"ACTION1": [-6.0, 2]}
    out = a._rprog_decide(_cands("ACTION1"))
    assert out is None
    assert a._rprog_fires == 0


# --- sem dados para os cands → None ---
def test_rprog_decide_none_when_no_data(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_RPROG="1")
    out = a._rprog_decide(_cands("ACTION1", "ACTION2"))
    assert out is None
    assert a._rprog_fires == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/rprog-modelfree && uv run pytest tests/causal/test_agent_rprog.py -k rprog_decide -v`
Expected: FAIL — `AttributeError: ... has no attribute '_rprog_decide'`.

- [ ] **Step 3: Add the `_rprog_decide` method**

In `agents/causal/agent.py`, add right after `_track_rprog`:

```python
    def _rprog_decide(self, cands):
        """B′: escolhe a ação de maior Δ médio de reward real (>0). Sem dados/positivo → None."""
        best_key, best_avg = None, 0.0
        for c in cands:
            row = self._rprog.get(c.key)
            if not row or row[1] == 0:
                continue
            avg = row[0] / row[1]
            if avg > best_avg:
                best_avg, best_key = avg, c.key
        if best_key is not None:
            self._rprog_fires += 1
        return best_key
```

- [ ] **Step 4: Wire it into the decision stack**

In `choose_action`, the navigate block is (`agent.py:226-229`):

```python
        if cand is None and self._nav_on:
            nk = navigate(scene, self._move)
            if nk is not None:
                cand = keymap.get(nk)
```

Right after it (before the `if cand is None and self._iw_on and cands:` block), insert:

```python
        if cand is None and self._rprog_on and cands:
            rk = self._rprog_decide(cands)    # B′: progresso model-free por reward real
            if rk is not None:
                cand = keymap.get(rk)
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/rprog-modelfree && uv run pytest tests/causal/test_agent_rprog.py -v`
Expected: PASS (8 tests total no arquivo).

- [ ] **Step 6: Run the full causal suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/rprog-modelfree && uv run pytest tests/causal -q`
Expected: PASS (default-off → pilha inalterada nos outros testes).

- [ ] **Step 7: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_rprog.py
git commit -m "feat: camada _rprog_decide (navigate→rprog→IW) — dirige pela evidência de reward real"
```

---

### Task 3: toggle `CAUSAL_RPROG` nos builders de notebook

**Files:**
- Modify: `kaggle/build_notebook.py` (bloco `.env`, ~linha 45) e `kaggle/build_offline_notebook.py` (~linha 27)
- Test: `tests/kaggle/test_build_offline_notebook.py` (append) e `tests/kaggle/test_build_notebook.py` (append)

**Interfaces:**
- Consumes: os blocos `.env` que já contêm `CAUSAL_ETA=1`/`CAUSAL_IW=1`.
- Produces: o `.env` gerado por ambos os builders contém `CAUSAL_RPROG=1`.

- [ ] **Step 1: Confirm the env-block shape of each builder**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/rprog-modelfree && grep -n "CAUSAL_IW\|OFFLINE_ENV\|_ENV =\|def build_notebook\|def read_sources" kaggle/build_notebook.py kaggle/build_offline_notebook.py`
Use the result to pick, for each test below, whether to assert against a module constant (like `OFFLINE_ENV`) or against the built notebook text. `build_offline_notebook.py` exposes `OFFLINE_ENV`; mirror whatever `build_notebook.py` exposes.

- [ ] **Step 2: Write the failing tests**

Append to `tests/kaggle/test_build_offline_notebook.py`:

```python
def test_offline_env_has_rprog():
    import kaggle.build_offline_notebook as b
    assert "CAUSAL_RPROG=1" in b.OFFLINE_ENV
```

Append to `tests/kaggle/test_build_notebook.py` (assert against the built notebook text so it is robust to how the env block is stored):

```python
def test_env_has_rprog():
    import kaggle.build_notebook as b
    nb = b.build_notebook(b.read_sources())
    text = "".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"]
    )
    assert "CAUSAL_RPROG=1" in text
```

If the exact `build_notebook(...)`/`read_sources()` signatures differ from this (confirm in Step 1), adjust the call to match — the assertion (`"CAUSAL_RPROG=1" in text`) stays the same.

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/rprog-modelfree && uv run pytest tests/kaggle/test_build_offline_notebook.py::test_offline_env_has_rprog tests/kaggle/test_build_notebook.py::test_env_has_rprog -v`
Expected: FAIL — `CAUSAL_RPROG=1` ausente.

- [ ] **Step 4: Add the flag to both builders**

In `kaggle/build_offline_notebook.py`, in the `.env` string block, after the `"CAUSAL_IW=1\n"` line, add:

```python
    "CAUSAL_RPROG=1\n"      # progresso model-free por reward real (Lever B')
```

In `kaggle/build_notebook.py`, after its `"CAUSAL_IW=1\n"` line, add the same:

```python
    "CAUSAL_RPROG=1\n"      # progresso model-free por reward real (Lever B')
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/rprog-modelfree && uv run pytest tests/kaggle -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/rprog-modelfree && uv run pytest tests/causal tests/kaggle -q`
Expected: PASS, count = 331 baseline + 5 (Task1) + 3 (Task2) + 2 (Task3) = 341.

- [ ] **Step 7: Commit**

```bash
git add kaggle/build_notebook.py kaggle/build_offline_notebook.py tests/kaggle/test_build_notebook.py tests/kaggle/test_build_offline_notebook.py
git commit -m "feat: CAUSAL_RPROG=1 no .env dos 2 builders (liga o progresso model-free)"
```

---

## Notes for the offline notebook

Após o merge, regenerar com `uv run python kaggle/build_offline_notebook.py`. Rodar **1 jogo por vez** (evita o gargalo #2 — reload do 32B por subprocesso que trava na troca de jogo). Validação real: observar `rprog_fires > 0` (a camada dirige) e se `levels_completed` sobe.
