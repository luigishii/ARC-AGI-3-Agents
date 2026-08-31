# Guarda Global Anti-Fixação Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quebrar runs da mesma ação (fixação) com um guarda que roda DEPOIS da pilha inteira: se a mesma key repete K vezes, sobrepõe por uma escolha de cobertura, pegando fixação de qualquer camada.

**Architecture:** Método `_antifix(cand, cands, keymap)` chamado no fim do `choose_action` (após o cand ser finalizado); conta repetições da key vs `_last_key`, e ao atingir `FIX_K` sobrepõe via `_cover_decide` (excluindo a key fixada); diagnóstico `fix_breaks`; toggle `CAUSAL_FIX`.

**Tech Stack:** Python 3.12, pytest. Só `agents/causal/agent.py` + `kaggle/build_notebook.py` + `kaggle/build_offline_notebook.py` + testes.

## Global Constraints

- Default-safe: guarda só sob `CAUSAL_FIX` (default off); decisão idêntica sem o toggle.
- Reusa `_cover_decide` (lever de cobertura, já em `agent.py`).
- Manter a suíte verde (baseline 363 na `main` `138ee39`; confirme com `pytest tests/causal tests/kaggle -q`).
- Não hardcodar `ARC_API_KEY` nem segredos.

---

### Task 1: estado + `_antifix` + wiring + diagnóstico

**Files:**
- Modify: `agents/causal/agent.py` — `_init_causal_state` (após `self._cover_on`, linha 107); novo método `_antifix` (perto de `_cover_decide`); `choose_action` (entre `return GameAction.RESET` na 273 e `action = cand.action` na 274); `phase2_stats` (após `cover_keys`, linha 485)
- Test: `tests/causal/test_agent_antifix.py` (create)

**Interfaces:**
- Consumes: `self._last_key`, `self._cover_decide`, `cands`, `keymap`.
- Produces: `self._fix_run`/`self._fix_breaks`/`self._fix_on`/`self._fix_k`; `_antifix(cand, cands, keymap) -> Candidate`; `phase2_stats` ganha `fix_breaks`.

- [ ] **Step 1: Write the failing tests**

Create `tests/causal/test_agent_antifix.py`:

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


def _keymap(cands):
    return {c.key: c for c in cands}


# --- quebra na K-ésima repetição, sobrepondo por candidata != key fixada ---
def test_antifix_breaks_after_k(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_FIX="1", CAUSAL_FIX_K="3")
    a._last_key = "A"
    cands = [_c("A"), _c("B")]
    km = _keymap(cands)
    out1 = a._antifix(_c("A"), cands, km)
    out2 = a._antifix(_c("A"), cands, km)
    out3 = a._antifix(_c("A"), cands, km)
    assert out1.key == "A" and out2.key == "A"
    assert out3.key == "B"                     # 3ª repetição -> sobrepõe
    assert a._fix_breaks == 1


# --- abaixo de K não quebra ---
def test_antifix_below_k(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_FIX="1", CAUSAL_FIX_K="3")
    a._last_key = "A"
    cands = [_c("A"), _c("B")]
    km = _keymap(cands)
    a._antifix(_c("A"), cands, km)
    out = a._antifix(_c("A"), cands, km)
    assert out.key == "A"
    assert a._fix_breaks == 0


# --- key diferente da anterior zera o run (nunca quebra alternando) ---
def test_antifix_diff_key_resets(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_FIX="1", CAUSAL_FIX_K="3")
    a._last_key = "A"
    cands = [_c("A"), _c("B")]
    km = _keymap(cands)
    for _ in range(5):
        out = a._antifix(_c("B"), cands, km)   # B != _last_key "A" -> run sempre 0
    assert out.key == "B"
    assert a._fix_breaks == 0


# --- sem alternativa (só a key fixada) não força ---
def test_antifix_no_alt(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_FIX="1", CAUSAL_FIX_K="3")
    a._last_key = "A"
    cands = [_c("A")]
    km = _keymap(cands)
    for _ in range(5):
        out = a._antifix(_c("A"), cands, km)
    assert out.key == "A"
    assert a._fix_breaks == 0


# --- off por default: nunca sobrepõe ---
def test_antifix_off_default(monkeypatch):
    a = _agent(monkeypatch)                     # CAUSAL_FIX não setado
    a._last_key = "A"
    cands = [_c("A"), _c("B")]
    km = _keymap(cands)
    for _ in range(5):
        out = a._antifix(_c("A"), cands, km)
    assert out.key == "A"
    assert a._fix_breaks == 0


# --- phase2_stats expõe fix_breaks ---
def test_phase2_has_fix_breaks(monkeypatch):
    a = _agent(monkeypatch)
    a._fix_breaks = 5
    assert a.phase2_stats()["fix_breaks"] == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/antifix-guard && uv run pytest tests/causal/test_agent_antifix.py -v`
Expected: FAIL — `AttributeError: ... '_antifix'` / `'_fix_breaks'` e `KeyError: 'fix_breaks'`.

- [ ] **Step 3: Init the fix state**

In `_init_causal_state`, right after `self._cover_on = os.environ.get("CAUSAL_COVER", "0") != "0"` (line 107), add:

```python
        self._fix_run = 0             # repetições consecutivas da MESMA key escolhida
        self._fix_breaks = 0          # diag: vezes que o guarda quebrou uma fixação
        self._fix_on = os.environ.get("CAUSAL_FIX", "0") != "0"
        self._fix_k = int(os.environ.get("CAUSAL_FIX_K", "3"))
```

- [ ] **Step 4: Add the `_antifix` method**

In `agents/causal/agent.py`, add right after `_cover_decide`:

```python
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
```

- [ ] **Step 5: Wire it into `choose_action`**

In `choose_action`, between `return GameAction.RESET` (line 273) and `action = cand.action` (line 274), insert:

```python
        cand = self._antifix(cand, cands, keymap)   # guarda global anti-fixação
```

- [ ] **Step 6: Add the phase2_stats key**

In `phase2_stats`, right after `"cover_keys": len(self._cover),` (line 485), insert:

```python
            "fix_breaks": self._fix_breaks,
```

- [ ] **Step 7: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/antifix-guard && uv run pytest tests/causal/test_agent_antifix.py -v`
Expected: PASS (6 tests).

- [ ] **Step 8: Run the full causal suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/antifix-guard && uv run pytest tests/causal -q`
Expected: PASS (default-off → decisão inalterada nos outros testes).

- [ ] **Step 9: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_antifix.py
git commit -m "feat: guarda global anti-fixação (_antifix quebra runs da mesma ação) + diag fix_breaks"
```

---

### Task 2: toggle `CAUSAL_FIX` nos builders

**Files:**
- Modify: `kaggle/build_notebook.py` (após `CAUSAL_COVER=1`, linha 47) e `kaggle/build_offline_notebook.py` (após `CAUSAL_COVER=1`, linha 29)
- Test: `tests/kaggle/test_build_offline_notebook.py` (append) e `tests/kaggle/test_build_notebook.py` (append)

**Interfaces:**
- Consumes: os blocos `.env` que já contêm `CAUSAL_COVER=1`.
- Produces: o `.env` gerado por ambos os builders contém `CAUSAL_FIX=1`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/kaggle/test_build_offline_notebook.py`:

```python
def test_offline_env_has_fix():
    import kaggle.build_offline_notebook as b
    assert "CAUSAL_FIX=1" in b.OFFLINE_ENV
```

Append to `tests/kaggle/test_build_notebook.py`:

```python
def test_env_has_fix():
    import kaggle.build_notebook as b
    assert "CAUSAL_FIX=1" in b.ENV
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/antifix-guard && uv run pytest tests/kaggle/test_build_offline_notebook.py::test_offline_env_has_fix tests/kaggle/test_build_notebook.py::test_env_has_fix -v`
Expected: FAIL — `CAUSAL_FIX=1` ausente.

- [ ] **Step 3: Add the flag to both builders**

In `kaggle/build_offline_notebook.py`, after the `"CAUSAL_COVER=1\n"` line (line 29), add:

```python
    "CAUSAL_FIX=1\n"        # guarda global anti-fixação
```

In `kaggle/build_notebook.py`, after its `"CAUSAL_COVER=1\n"` line (line 47), add the same:

```python
    "CAUSAL_FIX=1\n"        # guarda global anti-fixação
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/antifix-guard && uv run pytest tests/kaggle -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/antifix-guard && uv run pytest tests/causal tests/kaggle -q`
Expected: PASS (baseline 363 + 6 Task1 + 2 Task2 = 371).

- [ ] **Step 6: Commit**

```bash
git add kaggle/build_notebook.py kaggle/build_offline_notebook.py tests/kaggle/test_build_notebook.py tests/kaggle/test_build_offline_notebook.py
git commit -m "feat: CAUSAL_FIX=1 no .env dos 2 builders (liga o guarda anti-fixação)"
```

---

## Notes for the offline notebook

Após o merge, regenerar com `uv run python kaggle/build_offline_notebook.py`. Validação real (offline multi-jogo `OFFLINE_GAMES="vc33,ls20,tn36,sk48"`): observar `fix_breaks > 0` e se o `ls20` para de martelar ACTION4 (ações variadas) e se `levels_completed` sobe. Se quebrar sem cruzar → próximo lever = recência no rprog/η (causa upstream).
