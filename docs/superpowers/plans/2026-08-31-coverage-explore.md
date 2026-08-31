# Exploração por Cobertura + Anti-Fixação Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quebrar a fixação do fallback guloso com uma camada que escolhe sempre a candidata menos visitada (sweep sistemático das 36 células do grid + ações), pré-requisito pra achar a sequência que resolve o nível.

**Architecture:** `self._cover` conta ações tomadas (close-loop); `_cover_decide(cands)` na pilha (entre η e greedy, sob `CAUSAL_COVER`) escolhe a de menor contagem, desempatando por `has_object` e anti-repetição; diagnóstico `cover_keys`; toggle nos 2 builders.

**Tech Stack:** Python 3.12, pytest. Só `agents/causal/agent.py` + `kaggle/build_notebook.py` + `kaggle/build_offline_notebook.py` + testes.

## Global Constraints

- Default-safe: camada só sob `CAUSAL_COVER` (default off); pilha idêntica sem o toggle.
- Contagem só em transição decisão→decisão (o `else` que já protege `f_τ`/η — junto de `_track_rprog`).
- Manter a suíte verde (baseline na `main` `a604a44`; confirme com `pytest tests/causal tests/kaggle -q`).
- Não hardcodar `ARC_API_KEY` nem segredos.

---

### Task 1: estado + contagem + `_cover_decide` + wiring + diagnóstico

**Files:**
- Modify: `agents/causal/agent.py` — `_init_causal_state` (após `self._rprog_on`, linha 105); close-loop (após `self._track_rprog(scene)`, linha 166); novos métodos `_track_cover`/`_cover_decide`; pilha de decisão (após o bloco η, ~linha 256, antes do greedy); `phase2_stats` (após `reward_rejected`, linha 463)
- Test: `tests/causal/test_agent_cover.py` (create)

**Interfaces:**
- Consumes: `self._last_key`, `cands` (lista de `Candidate` com `.key`/`.has_object`).
- Produces: `self._cover: dict[str,int]`, `self._cover_on: bool`; `_track_cover() -> None`; `_cover_decide(cands) -> key`; `phase2_stats` ganha `cover_keys`.

- [ ] **Step 1: Write the failing tests**

Create `tests/causal/test_agent_cover.py`:

```python
from agents.causal.agent import CausalObjectAgent
from agents.causal.policy import Candidate


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def _c(key, has_object=False):
    return Candidate(None, None, None, key, has_object)


# --- varre: escolhe uma; depois de visitá-la, escolhe OUTRA (menos visitada) ---
def test_cover_decide_sweeps(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_COVER="1")
    cands = [_c("A"), _c("B"), _c("C")]
    first = a._cover_decide(cands)
    assert first in ("A", "B", "C")
    a._cover[first] = 1                      # marca como visitada
    second = a._cover_decide(cands)
    assert second != first                   # próxima é outra (menos visitada)


# --- desempate: has_object vem antes de vazio ---
def test_cover_decide_prefers_object(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_COVER="1")
    out = a._cover_decide([_c("empty", False), _c("obj", True)])
    assert out == "obj"


# --- anti-repetição: evita a última key em empate ---
def test_cover_decide_avoids_last(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_COVER="1")
    a._last_key = "P"
    out = a._cover_decide([_c("P", True), _c("Q", True)])
    assert out == "Q"


# --- contagem no close-loop: _track_cover incrementa a key da última ação ---
def test_track_cover_counts(monkeypatch):
    a = _agent(monkeypatch)
    a._last_key = "ACTION1"
    a._track_cover()
    a._track_cover()
    assert a._cover["ACTION1"] == 2


# --- sem last_key não conta ---
def test_track_cover_no_last(monkeypatch):
    a = _agent(monkeypatch)
    a._last_key = None
    a._track_cover()
    assert a._cover == {}


# --- phase2_stats expõe cover_keys ---
def test_phase2_has_cover_keys(monkeypatch):
    a = _agent(monkeypatch)
    a._cover = {"A": 3, "B": 1}
    assert a.phase2_stats()["cover_keys"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/coverage-explore && uv run pytest tests/causal/test_agent_cover.py -v`
Expected: FAIL — `AttributeError: ... '_cover_decide'` / `'_track_cover'` / `'_cover'` e `KeyError: 'cover_keys'`.

- [ ] **Step 3: Init the cover state**

In `_init_causal_state`, right after `self._rprog_on = os.environ.get("CAUSAL_RPROG", "0") != "0"` (line 105), add:

```python
        self._cover = {}              # action_key -> nº de vezes tomada (exploração por cobertura)
        self._cover_on = os.environ.get("CAUSAL_COVER", "0") != "0"
```

- [ ] **Step 4: Add `_track_cover` and `_cover_decide` methods**

In `agents/causal/agent.py`, add near `_rprog_decide` (e.g., right after it):

```python
    def _track_cover(self):
        """Conta a ação efetivamente tomada (p/ o sweep de cobertura). Só decisão→decisão."""
        if self._last_key is not None:
            self._cover[self._last_key] = self._cover.get(self._last_key, 0) + 1

    def _cover_decide(self, cands):
        """Exploração por cobertura: escolhe a candidata MENOS visitada. Desempate:
        has_object antes de vazio; evita repetir a última ação; senão ordem de cands."""
        def rank(c):
            return (self._cover.get(c.key, 0),
                    0 if c.has_object else 1,
                    1 if c.key == self._last_key else 0)
        return min(cands, key=rank).key
```

- [ ] **Step 5: Count in the close-loop**

In `choose_action`, right after `self._track_rprog(scene)` (line 166, same indentation, inside the decisão→decisão `else`), add:

```python
                self._track_cover()               # cobertura: conta a ação tomada
```

- [ ] **Step 6: Wire `_cover_decide` into the stack (between η and greedy)**

In `choose_action`, the η block ends with (~lines 254-257):

```python
        if cand is None and self._eta_on and cands:
            ek = self._eta_explore(cands)     # sonda a ação de linha mais ambígua (η alto)
            if ek is not None:
                cand = keymap.get(ek)
```

Right after it (before `if cand is None:` that calls `self._policy.decide`), insert:

```python
        if cand is None and self._cover_on and cands:
            ck = self._cover_decide(cands)    # exploração por cobertura (anti-fixação)
            if ck is not None:
                cand = keymap.get(ck)
```

- [ ] **Step 7: Add the phase2_stats key**

In `phase2_stats`, right after `"reward_rejected": self._reward_rejected,` (line 463), insert:

```python
            "cover_keys": len(self._cover),
```

- [ ] **Step 8: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/coverage-explore && uv run pytest tests/causal/test_agent_cover.py -v`
Expected: PASS (6 tests).

- [ ] **Step 9: Run the full causal suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/coverage-explore && uv run pytest tests/causal -q`
Expected: PASS (default-off → pilha inalterada nos outros testes).

- [ ] **Step 10: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_cover.py
git commit -m "feat: exploração por cobertura + anti-fixação (_cover_decide na pilha) + diag cover_keys"
```

---

### Task 2: toggle `CAUSAL_COVER` nos builders

**Files:**
- Modify: `kaggle/build_notebook.py` (após `CAUSAL_RPROG=1`, linha 46) e `kaggle/build_offline_notebook.py` (após `CAUSAL_RPROG=1`, linha 28)
- Test: `tests/kaggle/test_build_offline_notebook.py` (append) e `tests/kaggle/test_build_notebook.py` (append)

**Interfaces:**
- Consumes: os blocos `.env` que já contêm `CAUSAL_RPROG=1`.
- Produces: o `.env` gerado por ambos os builders contém `CAUSAL_COVER=1`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/kaggle/test_build_offline_notebook.py`:

```python
def test_offline_env_has_cover():
    import kaggle.build_offline_notebook as b
    assert "CAUSAL_COVER=1" in b.OFFLINE_ENV
```

Append to `tests/kaggle/test_build_notebook.py`:

```python
def test_env_has_cover():
    import kaggle.build_notebook as b
    assert "CAUSAL_COVER=1" in b.ENV
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/coverage-explore && uv run pytest tests/kaggle/test_build_offline_notebook.py::test_offline_env_has_cover tests/kaggle/test_build_notebook.py::test_env_has_cover -v`
Expected: FAIL — `CAUSAL_COVER=1` ausente.

- [ ] **Step 3: Add the flag to both builders**

In `kaggle/build_offline_notebook.py`, after the `"CAUSAL_RPROG=1\n"` line (line 28), add:

```python
    "CAUSAL_COVER=1\n"      # exploração por cobertura + anti-fixação
```

In `kaggle/build_notebook.py`, after its `"CAUSAL_RPROG=1\n"` line (line 46), add the same:

```python
    "CAUSAL_COVER=1\n"      # exploração por cobertura + anti-fixação
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/coverage-explore && uv run pytest tests/kaggle -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/coverage-explore && uv run pytest tests/causal tests/kaggle -q`
Expected: PASS (baseline + 6 Task1 + 2 Task2).

- [ ] **Step 6: Commit**

```bash
git add kaggle/build_notebook.py kaggle/build_offline_notebook.py tests/kaggle/test_build_notebook.py tests/kaggle/test_build_offline_notebook.py
git commit -m "feat: CAUSAL_COVER=1 no .env dos 2 builders (liga a exploração por cobertura)"
```

---

## Notes for the offline notebook

Após o merge, regenerar com `uv run python kaggle/build_offline_notebook.py`. Validação real (foco vc33: `OFFLINE_GAMES="vc33"`): observar `cover_keys` subir muito (varreu, não fixou) e se `levels_completed` sobe. Depois validar nos outros 3 (mecanismo geral, anti-overfit).
