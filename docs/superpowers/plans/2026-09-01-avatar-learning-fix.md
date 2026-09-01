# Consertar aprendizado do avatar — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Destravar o aprendizado do avatar sob ruído de HUD (`_moved_object` isola o mover rígido) e adiar a síntese da reward até o avatar ser conhecido — para que avatar-grounding + alvo-heurístico finalmente entrem no prompt.

**Tech Stack:** Python 3.12, pytest.

## Global Constraints

- Default-safe (sem toggle novo). Sem mudança em `_build_reward_prompt`, `_pick_target`, `state`, `accept_reward`, pilha.
- `Object`: `.id`, `.centroid=(row,col)`, `.shape_hash`, `.size`.
- Suíte verde (base 385 + novos).

---

### Task 1: `_moved_object` isola o mover rígido (`agents/causal/navigate.py`)

**Files:** Modify `agents/causal/navigate.py`; Test `tests/causal/test_navigate_rigid.py` (criar).

- [ ] **Step 1: Failing test** — criar `tests/causal/test_navigate_rigid.py`:

```python
from types import SimpleNamespace as NS

from agents.causal.navigate import _moved_object, MovementModel


def _o(oid, centroid, shape, size):
    return NS(id=oid, centroid=centroid, shape_hash=shape, size=size)


def _scene(objs):
    return NS(objects=objs)


def test_isolates_rigid_ignoring_shrinking_bar():
    prev = _scene([_o(1, (5, 5), "A", 9), _o(2, (0, 5), "BAR", 10)])
    curr = _scene([_o(1, (5, 10), "A", 9),   # avatar transladou rigido
                   _o(2, (0, 3), "bar2", 7)])  # barra encolheu+shiftou (nao-rigido)
    m = _moved_object(prev, curr)
    assert m is not None and m[0] == 1


def test_two_rigid_movers_is_none():
    prev = _scene([_o(1, (5, 5), "A", 9), _o(2, (0, 0), "B", 4)])
    curr = _scene([_o(1, (5, 10), "A", 9), _o(2, (0, 5), "B", 4)])
    assert _moved_object(prev, curr) is None


def test_zero_movers_is_none():
    prev = _scene([_o(1, (5, 5), "A", 9)])
    curr = _scene([_o(1, (5, 5), "A", 9)])
    assert _moved_object(prev, curr) is None


def test_single_rigid_mover_returned():
    prev = _scene([_o(1, (5, 5), "A", 9)])
    curr = _scene([_o(1, (7, 5), "A", 9)])
    m = _moved_object(prev, curr)
    assert m == (1, (2, 0))


def test_movement_model_learns_avatar_under_hud_noise():
    prev = _scene([_o(1, (5, 5), "A", 9), _o(2, (0, 5), "BAR", 10)])
    curr = _scene([_o(1, (5, 10), "A", 9), _o(2, (0, 3), "bar2", 7)])
    mm = MovementModel()
    mm.observe("ACTION1", prev, curr)
    assert mm.avatar_id() == 1
```

- [ ] **Step 2: Run fail** — `cd .../avatar-learning-fix && uv run pytest tests/causal/test_navigate_rigid.py -q` → algum FALHA (hoje `_moved_object` retorna None com a barra movendo junto).

- [ ] **Step 3: Implement** — substituir `_moved_object` em `navigate.py` por:

```python
def _moved_object(prev, curr):
    prevmap = {o.id: o for o in prev.objects}
    rigid = []
    for o in curr.objects:
        po = prevmap.get(o.id)
        if po is None:
            continue
        dr = round(o.centroid[0] - po.centroid[0])
        dc = round(o.centroid[1] - po.centroid[1])
        if (dr, dc) == (0, 0):
            continue
        if o.shape_hash == po.shape_hash and o.size == po.size:   # rigido = avatar
            rigid.append((o.id, (dr, dc)))
    return rigid[0] if len(rigid) == 1 else None
```

- [ ] **Step 4: Run pass** — `uv run pytest tests/causal/test_navigate_rigid.py -q` → 5 PASS.

- [ ] **Step 5: Commit** — `git add agents/causal/navigate.py tests/causal/test_navigate_rigid.py && git commit -m "fix: _moved_object isola o mover rigido (ignora barra de HUD que encolhe)"`

---

### Task 2: Adiar síntese da reward até avatar conhecido (`agents/causal/agent.py`)

**Files:** Modify `agents/causal/agent.py`; Test `tests/causal/test_agent_reward_defer.py` (criar).

**Interfaces:** novo método `_should_learn_reward(self) -> bool`; constante módulo `REWARD_DEFER_MAX`.

- [ ] **Step 1: Failing test** — criar `tests/causal/test_agent_reward_defer.py`:

```python
from agents.causal.agent import CausalObjectAgent, REWARD_DEFER_MAX


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_defer_until_deadline_when_no_avatar(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    a._reward_fn = None                      # sem avatar aprendido
    for _ in range(REWARD_DEFER_MAX):
        assert a._should_learn_reward() is False
    assert a._should_learn_reward() is True   # deadline atingido


def test_learn_immediately_when_avatar_known(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    a._reward_fn = None
    a._move.avatar_counts = {7: 3}            # avatar conhecido
    assert a._should_learn_reward() is True


def test_no_relearn_when_reward_exists(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    a._reward_fn = lambda s: (0.0, False)     # já aprendida
    assert a._should_learn_reward() is False
```

- [ ] **Step 2: Run fail** — `uv run pytest tests/causal/test_agent_reward_defer.py -q` → ImportError (`REWARD_DEFER_MAX`/`_should_learn_reward` não existem).

- [ ] **Step 3: Implement**

(a) Constante módulo perto de `TYPE_COOLDOWN` (linha ~33):
```python
REWARD_DEFER_MAX = 4      # ticks elegiveis a esperar o avatar antes de sintetizar a reward
```

(b) Em `_init_causal_state`, após `self._reward_rejected = 0` (linha ~157):
```python
        self._reward_defer = 0        # ticks ja esperados pelo avatar (gate de sintese)
```

(c) Novo método (perto de `_try_learn_reward`):
```python
    def _should_learn_reward(self) -> bool:
        """Gate: só sintetiza a reward quando o avatar já foi aprendido (grounding entra no
        prompt) OU após REWARD_DEFER_MAX ticks (deadline p/ jogos sem avatar/clique)."""
        if self._reward_fn is not None:
            return False
        if self._move.avatar_id() is not None or self._reward_defer >= REWARD_DEFER_MAX:
            return True
        self._reward_defer += 1
        return False
```

(d) Trocar o gate (linha ~281-282):
```python
            if self._should_learn_reward():
                self._try_learn_reward(scene)
```

- [ ] **Step 4: Run pass** — `uv run pytest tests/causal/test_agent_reward_defer.py -q` → 3 PASS.

- [ ] **Step 5: Full suite** — `uv run pytest tests/causal tests/kaggle -q` → verde (base 385 + 8 novos = 393). Conferir que testes existentes de `navigate`/reward não regridem.

- [ ] **Step 6: Commit** — `git add agents/causal/agent.py tests/causal/test_agent_reward_defer.py && git commit -m "feat: adiar sintese da reward ate avatar conhecido (_should_learn_reward + REWARD_DEFER_MAX)"`

---

### Task 3: Regenerar notebooks

- [ ] **Step 1:** `uv run python kaggle/build_notebook.py && uv run python kaggle/build_offline_notebook.py`
- [ ] **Step 2:** Verificar embed: `_should_learn_reward` no agent.py embutido dos 2 .ipynb (via base64 decode).
- [ ] **Step 3:** `git add kaggle/*.ipynb && git commit -m "build: regen notebooks com fix de aprendizado do avatar"`

---

## Self-Review

**Spec coverage:** _moved_object rígido (T1) ✓; reward defer com deadline (T2) ✓; testes 1-4 da spec cobertos ✓; notebooks (T3) ✓.

**Placeholder scan:** sem TBD; código completo. ✓

**Type consistency:** `_moved_object(prev,curr)->(id,(dr,dc))|None`; `_should_learn_reward(self)->bool`; `REWARD_DEFER_MAX` int usado no gate e no teste. ✓

**Regressão a vigiar:** o novo `_moved_object` retorna None p/ um único mover **não-rígido** (antes retornava-o). É intencional (não aprender HUD como avatar). O full-suite (T2 step 5) pega qualquer teste de navigate que dependa do comportamento antigo.
