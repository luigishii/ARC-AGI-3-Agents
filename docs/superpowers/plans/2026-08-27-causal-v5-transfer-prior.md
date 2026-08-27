# CausalObjectAgent v5 — Reuso inter-jogos (TransferPrior abstrato) · Plano

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dar ao agente um warm-start generalizável — enviesar a exploração de um jogo novo pras modalidades de interação (simples/clique-em-objeto/clique-em-vazio) historicamente produtivas, agregadas entre todos os jogos do run via um prior compartilhado.

**Architecture:** Novo módulo `agents/causal/transfer.py` (`abstract_feature`, `TransferPrior` thread-safe, singleton `shared_prior`). `policy.py` ganha termo v5 em `score`/`decide` sob parâmetro opcional `prior=None` (None = v4 idêntico). `agent.py` usa o singleton, o alimenta e o passa ao `decide`. `perception/hud/causal_model/novelty/instrumentation` não mudam.

**Tech Stack:** Python 3.12, numpy/stdlib puro (`threading`), pytest. Sem LLM/GPU. Kaggle-submittable.

## Global Constraints

- Numpy/stdlib puro; nenhuma dependência nova; nada de LLM/GPU.
- `W_PRIOR = 1.0`; `NEUTRAL_PRODUCTIVITY = 0.5`; termo v5 no score = `W_PRIOR * prior.productivity(abstract_feature(cand))`.
- Features abstratas (3, nunca game-específicas): `"simple"`, `"click_on_object"`, `"click_empty"`.
- `prior=None` em `score`/`decide` deve reproduzir EXATAMENTE o v4.
- `TransferPrior.observe`/`productivity` sob `threading.Lock` (threads paralelas do Swarm).
- Não alterar `perception.py`, `hud.py`, `causal_model.py`, `novelty.py`, `instrumentation.py`.
- Os 82 testes v1–v4 em `tests/causal/` devem seguir verdes.
- Testes que tocam o singleton chamam `reset_shared_prior()` no início (isolar estado global).

---

### Task 1: `transfer.py` — `abstract_feature` + `TransferPrior` + singleton

**Files:**
- Create: `agents/causal/transfer.py`
- Test: `tests/causal/test_transfer.py` (novo)

**Interfaces:**
- Consumes: `Candidate` de `agents/causal/policy.py` (campos `.action` com `.is_complex()`, `.has_object`).
- Produces:
  - `W_PRIOR = 1.0`, `NEUTRAL_PRODUCTIVITY = 0.5`
  - `abstract_feature(cand) -> str`
  - `class TransferPrior`: `observe(feature, effect_kind)`, `productivity(feature) -> float`, `to_dict()`, `from_dict(d)`; atributo `_counts: dict[str,[int,int]]`.
  - `shared_prior() -> TransferPrior`, `reset_shared_prior() -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_transfer.py
import threading
from arcengine import GameAction
from agents.causal.policy import Candidate
from agents.causal.transfer import (
    abstract_feature, TransferPrior, shared_prior, reset_shared_prior,
    NEUTRAL_PRODUCTIVITY,
)


def _simple():
    return Candidate(GameAction.ACTION1, None, None, "ACTION1", False)


def _click(has_object):
    return Candidate(GameAction.ACTION6, 5, 5, "ACTION6@cell=0,0", has_object)


def test_abstract_feature_buckets():
    assert abstract_feature(_simple()) == "simple"
    assert abstract_feature(_click(True)) == "click_on_object"
    assert abstract_feature(_click(False)) == "click_empty"


def test_productivity_neutral_without_data():
    p = TransferPrior()
    assert p.productivity("click_on_object") == NEUTRAL_PRODUCTIVITY


def test_productivity_reflects_counts():
    p = TransferPrior()
    p.observe("click_on_object", "moved")     # produtivo
    p.observe("click_on_object", "none")      # não
    p.observe("click_on_object", None)        # não
    assert p.productivity("click_on_object") == 1 / 3


def test_observe_is_threadsafe():
    p = TransferPrior()

    def worker():
        for _ in range(1000):
            p.observe("simple", "moved")

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert p._counts["simple"] == [4000, 4000]


def test_roundtrip_serialization():
    p = TransferPrior()
    p.observe("simple", "moved")
    p.observe("simple", "none")
    d = p.to_dict()
    p2 = TransferPrior.from_dict(d)
    assert p2.to_dict() == d
    assert p2.productivity("simple") == p.productivity("simple")


def test_singleton_identity_and_reset():
    reset_shared_prior()
    a = shared_prior()
    b = shared_prior()
    assert a is b
    reset_shared_prior()
    assert shared_prior() is not a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v5 && uv run pytest tests/causal/test_transfer.py -q`
Expected: FAIL (ModuleNotFoundError: `agents.causal.transfer`).

- [ ] **Step 3: Write minimal implementation**

```python
# agents/causal/transfer.py
from __future__ import annotations

import threading

W_PRIOR = 1.0
NEUTRAL_PRODUCTIVITY = 0.5


def abstract_feature(cand) -> str:
    if not cand.action.is_complex():
        return "simple"
    return "click_on_object" if cand.has_object else "click_empty"


class TransferPrior:
    def __init__(self):
        self._counts = {}          # feature -> [n_produtivo, n_total]
        self._lock = threading.Lock()

    def observe(self, feature, effect_kind) -> None:
        with self._lock:
            c = self._counts.setdefault(feature, [0, 0])
            c[1] += 1
            if effect_kind not in (None, "none"):
                c[0] += 1

    def productivity(self, feature) -> float:
        with self._lock:
            c = self._counts.get(feature)
            if not c or c[1] == 0:
                return NEUTRAL_PRODUCTIVITY
            return c[0] / c[1]

    def to_dict(self) -> dict:
        with self._lock:
            return {"counts": {k: list(v) for k, v in self._counts.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "TransferPrior":
        p = cls()
        p._counts = {k: list(v) for k, v in d.get("counts", {}).items()}
        return p


_SHARED = TransferPrior()


def shared_prior() -> TransferPrior:
    return _SHARED


def reset_shared_prior() -> None:
    global _SHARED
    _SHARED = TransferPrior()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v5 && uv run pytest tests/causal/test_transfer.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/transfer.py tests/causal/test_transfer.py
git commit -m "feat(causal): TransferPrior + abstract_feature + singleton compartilhado"
```

---

### Task 2: termo v5 no `Policy.score`/`decide` (param opcional `prior`)

**Files:**
- Modify: `agents/causal/policy.py` (`Policy.score`, `Policy.decide`)
- Test: `tests/causal/test_policy_prior.py` (novo)

**Interfaces:**
- Consumes: `TransferPrior.productivity`, `abstract_feature`, `W_PRIOR` (Task 1).
- Produces: `Policy.score(self, cand, model, seen_effects, budget_frac, novelty=None, prior=None)` e `Policy.decide(self, scene, model, available_actions, seen_effects, budget_frac, novelty=None, prior=None)`.

**Nota de import:** `transfer.py` importa `policy.Candidate`; para evitar ciclo em tempo de import, `policy.py` importa `abstract_feature`/`W_PRIOR` de `transfer` DENTRO do `score` (import local), não no topo do módulo.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_policy_prior.py
from arcengine import GameAction
from agents.causal.perception import Scene
from agents.causal.causal_model import CausalModel
from agents.causal.transfer import TransferPrior
from agents.causal.policy import Policy, Candidate


def _click(has_object, key="ACTION6@cell=0,0"):
    return Candidate(GameAction.ACTION6, 5, 5, key, has_object)


def test_prior_boosts_productive_feature():
    p = Policy(seed=0)
    model = CausalModel()
    prior = TransferPrior()
    # torna "click_on_object" muito produtivo e "click_empty" improdutivo
    for _ in range(9):
        prior.observe("click_on_object", "moved")
    prior.observe("click_on_object", "none")
    for _ in range(9):
        prior.observe("click_empty", "none")
    prior.observe("click_empty", "moved")
    s_obj = p.score(_click(True), model, set(), 0.0, prior=prior)
    s_empty = p.score(_click(False), model, set(), 0.0, prior=prior)
    assert s_obj > s_empty


def test_prior_none_reproduces_v4():
    p = Policy(seed=0)
    model = CausalModel()
    # sem prior nem novelty: chave inédita → +3 (caminho v4/v3), has_object=False
    assert p.score(_click(False), model, set(), 0.0) == 3.0


def test_prior_term_magnitude():
    p = Policy(seed=0)
    model = CausalModel()
    prior = TransferPrior()          # neutro 0.5 → termo +0.5
    # chave inédita: base 3.0 (eff None → +3) + prior 0.5 = 3.5 (has_object False)
    assert p.score(_click(False), model, set(), 0.0, prior=prior) == 3.5


def test_decide_accepts_prior_kwarg():
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    prior = TransferPrior()
    scene = Scene(objects=[], grid=None)
    cand = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0, prior=prior)
    assert cand is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v5 && uv run pytest tests/causal/test_policy_prior.py -q`
Expected: FAIL (`score`/`decide` não aceitam `prior`).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/policy.py`, atualizar `Policy.score` e `Policy.decide`:

```python
    def score(self, cand, model, seen_effects, budget_frac, novelty=None, prior=None) -> float:
        eff, conf = model.predict(cand.key)
        s = 0.0
        if model.is_progress(cand.key):
            s += 10.0 * (1 + (1 - budget_frac))
        if novelty is None:
            if eff is None:
                s += 3.0
            elif conf < 0.8:
                s += 1.5
            if eff is not None and eff.kind not in seen_effects:
                s += 0.5
        else:
            y = novelty.yield_estimate(cand.key)
            ctrl = conf if eff is not None else 1.0
            s += 3.0 * y * ctrl
        if eff is not None and eff.kind == "none":
            s -= 2.0
        if cand.has_object:
            s += 0.5
        if prior is not None:
            from .transfer import abstract_feature, W_PRIOR
            s += W_PRIOR * prior.productivity(abstract_feature(cand))
        return s

    def decide(self, scene, model, available_actions, seen_effects, budget_frac, novelty=None, prior=None):
        cands = candidates(scene, available_actions)
        if not cands:
            return None
        if self._rng.random() < self.epsilon:
            return self._rng.choice(cands)
        best, best_s = None, None
        for c in cands:
            sc = self.score(c, model, seen_effects, budget_frac, novelty=novelty, prior=prior)
            if best_s is None or sc > best_s:
                best, best_s = c, sc
        return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v5 && uv run pytest tests/causal/test_policy_prior.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/policy.py tests/causal/test_policy_prior.py
git commit -m "feat(causal): termo v5 no score/decide (TransferPrior productivity)"
```

---

### Task 3: wiring no `agent.py` + integração e regressão

**Files:**
- Modify: `agents/causal/agent.py`
- Test: `tests/causal/test_agent_prior.py` (novo)

**Interfaces:**
- Consumes: `shared_prior`, `abstract_feature`, `TransferPrior` (Task 1); `Policy.decide(..., prior=)` (Task 2).
- Produces: `CausalObjectAgent` com `self._prior = shared_prior()` e `self._last_feature`, alimentando o prior no fecha-loop e passando-o ao `decide`; RESET não zera.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_agent_prior.py
from arcengine import GameAction, GameState
from agents.causal.agent import CausalObjectAgent
from agents.causal.transfer import TransferPrior, shared_prior, reset_shared_prior


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


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_agent_uses_shared_prior_instance():
    reset_shared_prior()
    a = _agent()
    assert a._prior is shared_prior()
    assert isinstance(a._prior, TransferPrior)


def test_agent_feeds_prior_over_steps():
    reset_shared_prior()
    a = _agent()
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))     # fecha o loop da 1ª ação → observe
    total = sum(v[1] for v in a._prior._counts.values())
    assert total >= 1


def test_two_agents_share_same_prior():
    reset_shared_prior()
    a = _agent()
    b = _agent()
    assert a._prior is b._prior


def test_reset_does_not_wipe_prior():
    reset_shared_prior()
    a = _agent()
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))
    a._prior.observe("simple", "moved")
    before = sum(v[1] for v in a._prior._counts.values())
    a.choose_action([], _Frame(_grid(3), state=GameState.GAME_OVER))  # RESET
    after = sum(v[1] for v in a._prior._counts.values())
    assert after >= before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v5 && uv run pytest tests/causal/test_agent_prior.py -q`
Expected: FAIL (`a._prior` não existe).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/agent.py`:

(a) no topo, adicionar o import:

```python
from .transfer import shared_prior, abstract_feature
```

(b) em `_init_causal_state`, após `self._novelty = NoveltyModel()`:

```python
        self._prior = shared_prior()
        self._last_feature = None
```

(c) no bloco de fecha-loop, logo após `self._novelty.observe_transition(self._last_key, scene)`:

```python
            if self._last_feature is not None:
                self._prior.observe(self._last_feature, actual.kind)
```

(d) trocar a chamada do decide para passar o prior:

```python
        cand = self._policy.decide(
            scene, self._model, latest_frame.available_actions or [GameAction.ACTION1],
            self._seen_effects, budget_frac, novelty=self._novelty, prior=self._prior,
        )
```

(e) ao guardar o estado do passo (junto de `self._last_key = cand.key`), guardar a feature:

```python
        self._last_feature = abstract_feature(cand)
```

(NÃO resetar `self._prior`/`self._last_feature` no branch de RESET — o prior é
compartilhado e persistente; `self._last_feature` pode ser deixado como está,
mas não zere o prior.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v5 && uv run pytest tests/causal/test_agent_prior.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Rodar a suíte inteira (regressão)**

Run: `cd .claude/worktrees/causal-v5 && uv run pytest tests/causal/ -q`
Expected: PASS (todos — 82 v1–v4 + novos v5).

Nota: se um teste v1–v4 quebrar por vazamento do singleton (estado do prior
acumulado entre testes), NÃO enfraquecer o teste — adicionar `reset_shared_prior()`
no setup do teste afetado. Mas os testes v1–v4 não passam `prior` ao `score`, então
não devem ser afetados.

- [ ] **Step 6: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_prior.py
git commit -m "feat(causal): agent usa TransferPrior compartilhado (reuso inter-jogos); regressão verde"
```

---

## Fora de escopo (4b)

- Persistência em disco (load no init / save no cleanup) e pré-treino offline pro Kaggle.
- Warm-start dos modelos crus; features abstratas mais ricas.

## Validação ao vivo (pós-merge, fora do plano de código)

Rodar 2 jogos no MESMO run e conferir cross-pollination do prior compartilhado.
Como o singleton vive só dentro de um processo, um script curto que instancia
2 agentes e roda alguns passos em cada, então imprime `shared_prior()._counts`,
já evidencia contagens agregadas de features de ambos os jogos. Alternativamente
`uv run main.py --agent=causalobject` (sem `--game`) roda o swarm em todos os
jogos e o prior acumula de todos.
