# CausalObjectAgent v8 — Planejamento com forward-model · Plano

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dar ao agente um forward-model de transição de estado e um planner beam que persegue estados-fronteira (ou âncoras de meta), num agente híbrido plan-or-fallback.

**Architecture:** Novo `agents/causal/planning.py` (`TransitionModel` + `plan`). `agents/causal/agent.py` aprende transições e planeja a cada passo, caindo na policy gulosa (v7) quando o planner não tem sinal. Toggle `CAUSAL_PLAN`. Demais módulos não mudam.

**Tech Stack:** Python 3.12, numpy/stdlib puro (`math`, `os`), pytest. Sem LLM/GPU. Kaggle-submittable.

## Global Constraints

- Numpy/stdlib puro; nenhuma dependência nova; nada de LLM/GPU.
- `plan(start_sig, start_keys, tmodel, novelty, anchors, depth=3, beam=8) -> str|None`.
- Objetivo sem âncora = `novelty.novelty(sig)`; fronteira (`predict_next None`) = 1.0; com âncora = `-min dist`.
- `plan` retorna `None` se NENHUMA `start_key` tem transição conhecida (→ fallback guloso).
- Toggle `CAUSAL_PLAN` (env): default on; `CAUSAL_PLAN=0` reproduz o v7.
- Reusa `state_signature` de `agents/causal/novelty.py`; candidatos via `agents/causal/policy.candidates`.
- Não alterar `perception/hud/causal_model/policy/novelty/transfer/instrumentation`.
- Os 117 testes v1–v7 devem seguir verdes.

---

### Task 1: `TransitionModel` (`planning.py`)

**Files:**
- Create: `agents/causal/planning.py`
- Test: `tests/causal/test_planning.py` (novo)

**Interfaces:**
- Consumes: nada (dicts de strings).
- Produces: `class TransitionModel` com `observe(prev_sig, key, next_sig)`, `predict_next(sig, key) -> str|None`, `known_keys(sig) -> list[str]`, `to_dict()/from_dict(d)`; atributo `trans: dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_planning.py
from agents.causal.planning import TransitionModel


def test_observe_and_predict_modal():
    m = TransitionModel()
    m.observe("A", "k1", "s1")
    m.observe("A", "k1", "s1")
    m.observe("A", "k1", "s2")     # s1 é modal (2 vs 1)
    assert m.predict_next("A", "k1") == "s1"


def test_predict_unknown_is_none():
    m = TransitionModel()
    assert m.predict_next("A", "k1") is None


def test_known_keys():
    m = TransitionModel()
    m.observe("A", "k1", "s1")
    m.observe("A", "k2", "s2")
    assert set(m.known_keys("A")) == {"k1", "k2"}
    assert m.known_keys("Z") == []


def test_roundtrip_serialization():
    m = TransitionModel()
    m.observe("A", "k1", "s1")
    d = m.to_dict()
    m2 = TransitionModel.from_dict(d)
    assert m2.to_dict() == d
    assert m2.predict_next("A", "k1") == "s1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v8 && uv run pytest tests/causal/test_planning.py -q`
Expected: FAIL (ModuleNotFoundError: `agents.causal.planning`).

- [ ] **Step 3: Write minimal implementation**

```python
# agents/causal/planning.py
from __future__ import annotations

PLAN_DEPTH = 3
PLAN_BEAM = 8


class TransitionModel:
    def __init__(self):
        self.trans = {}    # sig -> {key -> {next_sig -> count}}

    def observe(self, prev_sig, key, next_sig) -> None:
        d = self.trans.setdefault(prev_sig, {}).setdefault(key, {})
        d[next_sig] = d.get(next_sig, 0) + 1

    def predict_next(self, sig, key):
        d = self.trans.get(sig, {}).get(key)
        if not d:
            return None
        return max(d.items(), key=lambda kv: kv[1])[0]

    def known_keys(self, sig):
        return list(self.trans.get(sig, {}).keys())

    def to_dict(self) -> dict:
        return {s: {k: dict(nn) for k, nn in kk.items()}
                for s, kk in self.trans.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionModel":
        m = cls()
        m.trans = {s: {k: dict(nn) for k, nn in kk.items()}
                   for s, kk in d.items()}
        return m
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v8 && uv run pytest tests/causal/test_planning.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/planning.py tests/causal/test_planning.py
git commit -m "feat(causal): TransitionModel (forward-model de estado, serializável)"
```

---

### Task 2: `plan` — beam search (fronteira/novidade ou âncora)

**Files:**
- Modify: `agents/causal/planning.py` (funções `_sig_dist`, `_terminal_score`, `plan`)
- Test: `tests/causal/test_plan.py` (novo)

**Interfaces:**
- Consumes: `TransitionModel` (Task 1); um objeto com `.novelty(sig) -> float` (o `NoveltyModel` de `agents/causal/novelty.py`).
- Produces: `plan(start_sig, start_keys, tmodel, novelty, anchors, depth=PLAN_DEPTH, beam=PLAN_BEAM) -> str|None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_plan.py
from agents.causal.planning import plan, TransitionModel
from agents.causal.novelty import NoveltyModel


def test_plan_returns_none_without_data():
    m = TransitionModel()
    nov = NoveltyModel()
    assert plan("A", ["k1", "k2"], m, nov, []) is None


def test_plan_prefers_deeper_novel_state():
    m = TransitionModel()
    m.observe("A", "k1", "s1")
    m.observe("A", "k2", "s2")
    m.observe("s1", "k3", "s3")        # via k1 chega-se a s3 (2 passos)
    nov = NoveltyModel()
    for _ in range(50):
        nov.visit("s2")                # s2 muito visitado → baixa novidade
    # s3 nunca visitado → alta novidade; plano via k1 (A→s1→s3) vence via k2 (A→s2)
    assert plan("A", ["k1", "k2"], m, nov, [], depth=3) == "k1"


def test_plan_goal_directed_with_anchor():
    m = TransitionModel()
    m.observe("A", "k1", "3,0,0")      # k1 chega exatamente na âncora
    m.observe("A", "k2", "9,9,9")
    nov = NoveltyModel()
    assert plan("A", ["k1", "k2"], m, nov, ["3,0,0"], depth=2) == "k1"


def test_plan_frontier_is_attractive_no_anchor():
    m = TransitionModel()
    m.observe("A", "k1", "s1")         # k1 conhecido → estado s1
    # k2 é inédito a partir de A → fronteira (score 1.0)
    nov = NoveltyModel()
    for _ in range(50):
        nov.visit("s1")                # s1 saturado (baixa novidade)
    assert plan("A", ["k1", "k2"], m, nov, [], depth=2) == "k2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v8 && uv run pytest tests/causal/test_plan.py -q`
Expected: FAIL (`plan` inexistente).

- [ ] **Step 3: Write minimal implementation**

Adicionar em `agents/causal/planning.py`:

```python
def _sig_dist(a, b):
    sa = set(a.split(";")) if a else set()
    sb = set(b.split(";")) if b else set()
    return len(sa ^ sb)


def _terminal_score(sig, frontier, novelty, anchors):
    if anchors:
        if frontier:
            return -0.5                     # desconhecido: entre "na âncora" (0) e "longe"
        return -min(_sig_dist(sig, a) for a in anchors)
    if frontier:
        return 1.0
    return novelty.novelty(sig)


def plan(start_sig, start_keys, tmodel, novelty, anchors,
         depth=PLAN_DEPTH, beam=PLAN_BEAM):
    if not start_keys:
        return None
    # exige ao menos uma transição conhecida a partir do estado atual
    if all(tmodel.predict_next(start_sig, k) is None for k in start_keys):
        return None

    # nó = (first_key, sig_atual, frontier?)
    nodes = []
    for k in start_keys:
        nxt = tmodel.predict_next(start_sig, k)
        nodes.append((k, nxt if nxt is not None else start_sig, nxt is None))

    def score(node):
        _, sig, frontier = node
        return _terminal_score(sig, frontier, novelty, anchors)

    for _ in range(1, depth):
        nodes.sort(key=score, reverse=True)
        nodes = nodes[:beam]
        nxt_nodes = []
        for (first, sig, frontier) in nodes:
            keys = [] if frontier else tmodel.known_keys(sig)
            if not keys:
                nxt_nodes.append((first, sig, frontier))    # terminal (fronteira ou beco)
                continue
            for k in keys:
                nn = tmodel.predict_next(sig, k)
                nxt_nodes.append((first, nn if nn is not None else sig, nn is None))
        nodes = nxt_nodes

    return max(nodes, key=score)[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v8 && uv run pytest tests/causal/test_plan.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/planning.py tests/causal/test_plan.py
git commit -m "feat(causal): plan() beam (fronteira/novidade ou âncora)"
```

---

### Task 3: wiring no `agent.py` (híbrido plan-or-fallback) + regressão

**Files:**
- Modify: `agents/causal/agent.py`
- Test: `tests/causal/test_agent_planning.py` (novo)

**Interfaces:**
- Consumes: `TransitionModel`, `plan` (Tasks 1-2); `candidates` (policy); `state_signature` (novelty).
- Produces: `CausalObjectAgent` com `self._tmodel`, `self._last_sig`, `self._plan_on`; usa `plan` quando há sinal, senão `policy.decide`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_agent_planning.py
from arcengine import GameAction, GameState
from agents.causal.agent import CausalObjectAgent
from agents.causal.planning import TransitionModel


class _Frame:
    def __init__(self, frame, state=GameState.NOT_FINISHED, levels=0):
        self.frame = frame
        self.state = state
        self.levels_completed = levels
        self.available_actions = [GameAction.ACTION1]
        self.full_reset = False


def _grid(v):
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = v
    return [g]


def _agent(monkeypatch, plan_env=None):
    if plan_env is None:
        monkeypatch.delenv("CAUSAL_PLAN", raising=False)
    else:
        monkeypatch.setenv("CAUSAL_PLAN", plan_env)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._cleanup = False
    a._init_causal_state()
    return a


def test_agent_has_transition_model(monkeypatch):
    a = _agent(monkeypatch)
    assert isinstance(a._tmodel, TransitionModel)
    assert a._plan_on is True


def test_agent_learns_transitions(monkeypatch):
    a = _agent(monkeypatch)
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))       # fecha loop → observa 1 transição
    assert len(a._tmodel.trans) >= 1


def test_plan_disabled_reproduces_v7(monkeypatch):
    a = _agent(monkeypatch, plan_env="0")
    assert a._plan_on is False
    act = a.choose_action([], _Frame(_grid(3)))   # não estoura; retorna ação
    assert act is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v8 && uv run pytest tests/causal/test_agent_planning.py -q`
Expected: FAIL (`a._tmodel`/`a._plan_on` inexistentes).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/agent.py`:

(a) no topo, importar `plan`/`TransitionModel` e `candidates`:

```python
from .planning import TransitionModel, plan
```
e trocar `from .policy import Policy` por:
```python
from .policy import Policy, candidates
```

(b) em `_init_causal_state`, após `self._novelty = NoveltyModel()`:

```python
        self._tmodel = TransitionModel()
        self._last_sig = None
        self._plan_on = os.environ.get("CAUSAL_PLAN", "1") != "0"
```

(c) no fecha-loop, após `self._novelty.observe_transition(self._last_key, scene)`:

```python
            cur_sig = state_signature(scene)
            if self._last_sig is not None and self._last_key is not None:
                self._tmodel.observe(self._last_sig, self._last_key, cur_sig)
```

(d) substituir a chamada `cand = self._policy.decide(...)` por:

```python
        cands = candidates(scene, latest_frame.available_actions or [GameAction.ACTION1])
        cand = None
        if self._plan_on and cands:
            planned = plan(state_signature(scene), [c.key for c in cands],
                           self._tmodel, self._novelty, self._novelty.goal_anchors)
            if planned is not None:
                cand = {c.key: c for c in cands}.get(planned)
        if cand is None:
            cand = self._policy.decide(
                scene, self._model, latest_frame.available_actions or [GameAction.ACTION1],
                self._seen_effects, budget_frac, novelty=self._novelty, prior=self._prior,
            )
```

(e) guardar `self._last_sig` logo após `self._last_key = cand.key`:

```python
        self._last_sig = state_signature(scene)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v8 && uv run pytest tests/causal/test_agent_planning.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Rodar a suíte inteira (regressão)**

Run: `cd .claude/worktrees/causal-v8 && uv run pytest tests/causal tests/kaggle -q`
Expected: PASS (todos — 117 v1–v7 + novos v8). (`tests/unit/` do harness já falha na base — não rodar.)

- [ ] **Step 6: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_planning.py
git commit -m "feat(causal): agente híbrido plan-or-fallback (forward-model); regressão verde"
```

---

## Fora de escopo

- Persistência do `TransitionModel` em disco.
- Objetivos mais sofisticados (empowerment).

## Validação ao vivo (pós-merge, fora do plano de código)

Rodar 1+ jogo com `CAUSAL_MAX_ACTIONS≈2000` e o prior, comparando `CAUSAL_PLAN=1`
(padrão) vs `CAUSAL_PLAN=0` — a pergunta real: **cruza ≥1 nível?** Se não,
evidência de que o approach A não basta e o próximo passo é B (heurística por
jogo) ou C (exploit).
