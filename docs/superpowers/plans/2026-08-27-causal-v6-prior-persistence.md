# CausalObjectAgent v6 — Persistência do TransferPrior · Plano

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persistir o `TransferPrior` em disco (JSON atômico) com carga única no singleton compartilhado, para warm-start entre runs e prior pré-treinado read-only no Kaggle.

**Architecture:** `agents/causal/transfer.py` ganha `merge`, `save_prior`, `load_prior`, `load_shared_once` e `DEFAULT_PRIOR_PATH`; `reset_shared_prior` passa a zerar o flag de carga. `agents/causal/agent.py` carrega no init e salva no cleanup só sob env `CAUSAL_PRIOR_SAVE`. Demais módulos não mudam.

**Tech Stack:** Python 3.12, numpy/stdlib puro (`json`, `os`, `threading`), pytest. Sem LLM/GPU. Kaggle-submittable.

## Global Constraints

- Numpy/stdlib puro; nenhuma dependência nova; nada de LLM/GPU.
- `DEFAULT_PRIOR_PATH = "agents/causal/prior.json"`.
- Escrita atômica: temp `path+".tmp"` + `os.replace`.
- Escrita gated: só salva se `os.environ.get("CAUSAL_PRIOR_SAVE")`; caminho via `os.environ.get("CAUSAL_PRIOR", DEFAULT_PRIOR_PATH)`.
- Load funde no singleton **uma vez por processo** (`_load_lock` + `_loaded`).
- Save NÃO refunde o disco (evita dupla-contagem; acúmulo cross-run vem do load).
- `reset_shared_prior()` zera o singleton **e** `_loaded` (isolamento de teste).
- Não alterar `policy.py`, `perception.py`, `hud.py`, `causal_model.py`, `novelty.py`, `instrumentation.py`.
- Os 96 testes v1–v5 em `tests/causal/` devem seguir verdes.
- Escopo: SÓ o mecanismo; NÃO gerar/commitar um `prior.json` real.

---

### Task 1: `transfer.py` — `merge`, `save_prior`, `load_prior`

**Files:**
- Modify: `agents/causal/transfer.py` (imports `json`/`os`; `DEFAULT_PRIOR_PATH`; `TransferPrior.merge`; `save_prior`; `load_prior`)
- Test: `tests/causal/test_transfer_persistence.py` (novo)

**Interfaces:**
- Consumes: `TransferPrior` (v5) com `_counts`, `to_dict`, `from_dict`, `_lock`.
- Produces:
  - `DEFAULT_PRIOR_PATH = "agents/causal/prior.json"`
  - `TransferPrior.merge(other: TransferPrior) -> None`
  - `save_prior(prior, path) -> None` (atômico)
  - `load_prior(path) -> TransferPrior | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_transfer_persistence.py
import os
import json
from agents.causal.transfer import (
    TransferPrior, save_prior, load_prior, DEFAULT_PRIOR_PATH,
)


def _prior(counts):
    p = TransferPrior()
    p._counts = {k: list(v) for k, v in counts.items()}
    return p


def test_default_prior_path():
    assert DEFAULT_PRIOR_PATH == "agents/causal/prior.json"


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "prior.json")
    p = _prior({"click_on_object": [8, 10], "simple": [2, 2]})
    save_prior(p, path)
    assert os.path.exists(path)
    p2 = load_prior(path)
    assert p2.to_dict() == p.to_dict()
    assert p2.productivity("click_on_object") == 0.8


def test_save_is_atomic_no_tmp_left(tmp_path):
    path = str(tmp_path / "prior.json")
    save_prior(_prior({"simple": [1, 1]}), path)
    assert not os.path.exists(path + ".tmp")
    with open(path) as f:
        json.load(f)                      # JSON válido


def test_save_creates_missing_dir(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "prior.json")
    save_prior(_prior({"simple": [1, 1]}), path)
    assert os.path.exists(path)


def test_load_missing_returns_none(tmp_path):
    assert load_prior(str(tmp_path / "nope.json")) is None
    assert load_prior("") is None


def test_merge_accumulates():
    a = _prior({"simple": [1, 2], "click_empty": [0, 3]})
    b = _prior({"simple": [3, 4], "click_on_object": [5, 5]})
    a.merge(b)
    d = a.to_dict()["counts"]
    assert d["simple"] == [4, 6]
    assert d["click_empty"] == [0, 3]
    assert d["click_on_object"] == [5, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v6 && uv run pytest tests/causal/test_transfer_persistence.py -q`
Expected: FAIL (ImportError: `save_prior`/`load_prior`/`DEFAULT_PRIOR_PATH` inexistentes).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/transfer.py`: acrescentar `import json` e `import os` ao topo (junto de `import threading`); adicionar a constante e as funções, e o método `merge` dentro de `TransferPrior`.

Constante (após `NEUTRAL_PRODUCTIVITY = 0.5`):

```python
DEFAULT_PRIOR_PATH = "agents/causal/prior.json"
```

Método dentro de `class TransferPrior` (após `from_dict`):

```python
    def merge(self, other) -> None:
        for feat, (np_, nt) in other.to_dict()["counts"].items():
            with self._lock:
                c = self._counts.setdefault(feat, [0, 0])
                c[0] += np_
                c[1] += nt
```

Funções de módulo (após a classe, antes do bloco do singleton `_SHARED`):

```python
def save_prior(prior, path) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prior.to_dict(), f)
    os.replace(tmp, path)


def load_prior(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return TransferPrior.from_dict(json.load(f))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v6 && uv run pytest tests/causal/test_transfer_persistence.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/transfer.py tests/causal/test_transfer_persistence.py
git commit -m "feat(causal): save_prior/load_prior atômicos + TransferPrior.merge"
```

---

### Task 2: `load_shared_once` + `reset_shared_prior` zera o flag

**Files:**
- Modify: `agents/causal/transfer.py` (`_load_lock`, `_loaded`, `load_shared_once`; `reset_shared_prior`)
- Test: `tests/causal/test_transfer_load_once.py` (novo)

**Interfaces:**
- Consumes: `save_prior`, `load_prior`, `shared_prior`, `TransferPrior.merge` (Task 1 + v5).
- Produces: `load_shared_once(path) -> None`; `reset_shared_prior()` passa a zerar `_loaded`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_transfer_load_once.py
from agents.causal.transfer import (
    TransferPrior, save_prior, shared_prior, reset_shared_prior, load_shared_once,
)


def _seed_file(tmp_path):
    path = str(tmp_path / "prior.json")
    p = TransferPrior()
    p._counts = {"click_on_object": [9, 10]}   # produtividade 0.9
    save_prior(p, path)
    return path


def test_load_shared_once_merges_into_singleton(tmp_path):
    reset_shared_prior()
    path = _seed_file(tmp_path)
    load_shared_once(path)
    assert shared_prior().productivity("click_on_object") == 0.9


def test_load_shared_once_runs_only_once(tmp_path):
    reset_shared_prior()
    path = _seed_file(tmp_path)
    load_shared_once(path)
    load_shared_once(path)                       # 2ª vez: no-op (não duplica)
    assert shared_prior()._counts["click_on_object"] == [9, 10]


def test_reset_allows_reload(tmp_path):
    reset_shared_prior()
    path = _seed_file(tmp_path)
    load_shared_once(path)
    reset_shared_prior()                         # zera singleton E _loaded
    load_shared_once(path)
    assert shared_prior()._counts["click_on_object"] == [9, 10]


def test_load_shared_once_missing_file_is_noop(tmp_path):
    reset_shared_prior()
    load_shared_once(str(tmp_path / "nope.json"))
    assert shared_prior()._counts == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v6 && uv run pytest tests/causal/test_transfer_load_once.py -q`
Expected: FAIL (`load_shared_once` inexistente).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/transfer.py`, substituir o bloco do singleton (`_SHARED`,
`shared_prior`, `reset_shared_prior`) por:

```python
_SHARED = TransferPrior()
_load_lock = threading.Lock()
_loaded = False


def shared_prior() -> TransferPrior:
    return _SHARED


def reset_shared_prior() -> None:
    global _SHARED, _loaded
    _SHARED = TransferPrior()
    _loaded = False


def load_shared_once(path) -> None:
    global _loaded
    with _load_lock:
        if _loaded:
            return
        _loaded = True
        disk = load_prior(path)
        if disk is not None:
            shared_prior().merge(disk)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v6 && uv run pytest tests/causal/test_transfer_load_once.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/transfer.py tests/causal/test_transfer_load_once.py
git commit -m "feat(causal): load_shared_once (carga única) + reset zera _loaded"
```

---

### Task 3: wiring no `agent.py` (load no init + save gated no cleanup) + regressão

**Files:**
- Modify: `agents/causal/agent.py`
- Test: `tests/causal/test_agent_persistence.py` (novo)

**Interfaces:**
- Consumes: `load_shared_once`, `save_prior`, `DEFAULT_PRIOR_PATH` (Tasks 1-2); `shared_prior` (v5).
- Produces: `CausalObjectAgent._init_causal_state` carrega o prior; `CausalObjectAgent.cleanup(scorecard=None)` salva sob flag.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_agent_persistence.py
import os
from agents.causal.agent import CausalObjectAgent
from agents.causal.transfer import (
    TransferPrior, save_prior, reset_shared_prior, load_prior,
)


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._cleanup = False               # neutraliza o cleanup do base (sem API)
    a._init_causal_state()
    return a


def _seed(tmp_path):
    path = str(tmp_path / "prior.json")
    p = TransferPrior()
    p._counts = {"click_on_object": [9, 10]}
    save_prior(p, path)
    return path


def test_init_loads_prior_from_env(tmp_path, monkeypatch):
    reset_shared_prior()
    path = _seed(tmp_path)
    monkeypatch.setenv("CAUSAL_PRIOR", path)
    a = _agent()
    assert a._prior.productivity("click_on_object") == 0.9


def test_cleanup_saves_when_flag_set(tmp_path, monkeypatch):
    reset_shared_prior()
    path = str(tmp_path / "out.json")
    monkeypatch.setenv("CAUSAL_PRIOR", path)
    monkeypatch.setenv("CAUSAL_PRIOR_SAVE", "1")
    a = _agent()
    a._prior.observe("simple", "moved")
    a.cleanup()
    assert load_prior(path) is not None
    assert load_prior(path)._counts.get("simple") == [1, 1]


def test_cleanup_does_not_save_without_flag(tmp_path, monkeypatch):
    reset_shared_prior()
    path = str(tmp_path / "out.json")
    monkeypatch.setenv("CAUSAL_PRIOR", path)
    monkeypatch.delenv("CAUSAL_PRIOR_SAVE", raising=False)
    a = _agent()
    a._prior.observe("simple", "moved")
    a.cleanup()
    assert not os.path.exists(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v6 && uv run pytest tests/causal/test_agent_persistence.py -q`
Expected: FAIL (init não carrega env; `cleanup` não salva).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/agent.py`:

(a) trocar o import do `transfer`:

```python
from .transfer import shared_prior, abstract_feature, load_shared_once, DEFAULT_PRIOR_PATH
```

(b) em `_init_causal_state`, logo após `self._prior = shared_prior()`:

```python
        load_shared_once(os.environ.get("CAUSAL_PRIOR", DEFAULT_PRIOR_PATH))
```

(c) adicionar o override de `cleanup` como método da classe (p.ex. logo após
`is_done`):

```python
    def cleanup(self, scorecard=None):
        if os.environ.get("CAUSAL_PRIOR_SAVE"):
            from .transfer import save_prior
            save_prior(self._prior, os.environ.get("CAUSAL_PRIOR", DEFAULT_PRIOR_PATH))
        super().cleanup(scorecard)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v6 && uv run pytest tests/causal/test_agent_persistence.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Rodar a suíte inteira (regressão)**

Run: `cd .claude/worktrees/causal-v6 && uv run pytest tests/causal/ -q`
Expected: PASS (todos — 96 v1–v5 + novos v6).

Nota: o init agora chama `load_shared_once` com o caminho default
`agents/causal/prior.json` — que NÃO existe (não geramos o arquivo) →
`load_prior` retorna `None` e é no-op, então não afeta os testes v1–v5. Se
necessário, não enfraquecer testes — apontar `CAUSAL_PRIOR` para um arquivo
inexistente no setup.

- [ ] **Step 6: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_persistence.py
git commit -m "feat(causal): agent carrega prior no init e salva no cleanup (gated); regressão verde"
```

---

## Fora de escopo

- Gerar/commitar um `prior.json` pré-treinado (treino offline — depois/Fase 6).
- Merge-com-disco no save; lock de arquivo cross-processo.

## Validação (pós-merge, fora do plano de código)

Rodar 2 processos em sequência com `CAUSAL_PRIOR_SAVE=1` e o mesmo
`CAUSAL_PRIOR=/tmp/x.json` (via um script curto que instancia agentes, alimenta
o prior e chama `cleanup`), conferindo que as contagens **acumulam** entre os
dois (persistência cross-run).
