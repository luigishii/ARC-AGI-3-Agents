# CausalObjectAgent v4 — Modelo de objetivo (novidade-de-estado) · Plano

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trocar o motor de exploração de novidade-de-ação para novidade-de-estado controlável, dando ao agente um sinal de progresso intrínseco cold-start-safe.

**Architecture:** Novo módulo `agents/causal/novelty.py` (`state_signature`, `NoveltyModel`). `policy.py` ganha caminho v4 em `score`/`decide` sob parâmetro opcional `novelty=None` (None = comportamento v3 idêntico). `agent.py` instancia/atualiza/serializa o `NoveltyModel`. `perception/hud/causal_model/instrumentation` não mudam.

**Tech Stack:** Python 3.12, numpy/stdlib puro (`math`), pytest. Sem LLM/GPU. Kaggle-submittable.

## Global Constraints

- Numpy/stdlib puro; nenhuma dependência nova; nada de LLM/GPU.
- `OPTIMISTIC_YIELD = 1.0`; `novelty(sig) = 1/√(count+1)`; termo v4 no score = `3.0 * yield_estimate(key) * ctrl`, `ctrl = conf if eff is not None else 1.0`.
- Assinatura de estado: por objeto `(cor, gx, gy)` com `(gx,gy)=cell_of(centroid_col, centroid_row)`, ordenada, `";"`-join; cena vazia → `""`.
- `novelty=None` em `score`/`decide` deve reproduzir EXATAMENTE o v3.
- Não alterar `perception.py`, `hud.py`, `causal_model.py`, `instrumentation.py`.
- Os 65 testes v1–v3 em `tests/causal/` devem seguir verdes.
- Convenção de eixos: `x`=coluna=`centroid[1]`, `y`=linha=`centroid[0]`.

---

### Task 1: `novelty.py` — assinatura de estado

**Files:**
- Create: `agents/causal/novelty.py`
- Test: `tests/causal/test_novelty.py` (novo)

**Interfaces:**
- Consumes: `cell_of` de `agents/causal/policy.py`; `Scene`/`Object` de `agents/causal/perception.py`.
- Produces: `state_signature(scene) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_novelty.py
from agents.causal.perception import Scene, Object
from agents.causal.novelty import state_signature


def _obj(cells, color=3):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return Object(color, cset, bbox, centroid, len(cells), "h")


def test_empty_scene_signature_is_empty_string():
    assert state_signature(Scene(objects=[], grid=None)) == ""


def test_same_config_same_signature():
    a = Scene(objects=[_obj([(5, 5)])], grid=None)
    b = Scene(objects=[_obj([(5, 5)])], grid=None)
    assert state_signature(a) == state_signature(b)


def test_object_moved_to_other_cell_changes_signature():
    a = Scene(objects=[_obj([(5, 5)])], grid=None)      # célula (0,0)
    b = Scene(objects=[_obj([(50, 50)])], grid=None)    # célula (4,4)
    assert state_signature(a) != state_signature(b)


def test_signature_order_independent():
    o1 = _obj([(5, 5)], color=3)
    o2 = _obj([(50, 50)], color=4)
    a = Scene(objects=[o1, o2], grid=None)
    b = Scene(objects=[o2, o1], grid=None)
    assert state_signature(a) == state_signature(b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v4 && uv run pytest tests/causal/test_novelty.py -q`
Expected: FAIL (ModuleNotFoundError: `agents.causal.novelty`).

- [ ] **Step 3: Write minimal implementation**

```python
# agents/causal/novelty.py
from __future__ import annotations

import math

from .policy import cell_of

OPTIMISTIC_YIELD = 1.0


def state_signature(scene) -> str:
    parts = []
    for o in scene.objects:
        gx, gy = cell_of(int(round(o.centroid[1])), int(round(o.centroid[0])))
        parts.append((o.color, gx, gy))
    return ";".join(f"{c},{gx},{gy}" for (c, gx, gy) in sorted(parts))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v4 && uv run pytest tests/causal/test_novelty.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/novelty.py tests/causal/test_novelty.py
git commit -m "feat(causal): state_signature (config de objetos)"
```

---

### Task 2: `NoveltyModel` — contagem, novidade, yield, âncora, serialização

**Files:**
- Modify: `agents/causal/novelty.py` (adicionar classe `NoveltyModel`)
- Test: `tests/causal/test_novelty_model.py` (novo)

**Interfaces:**
- Consumes: `state_signature`, `OPTIMISTIC_YIELD` (Task 1).
- Produces: `class NoveltyModel` com `count(sig)`, `novelty(sig)`, `visit(sig)`, `observe_transition(key, curr_scene)`, `yield_estimate(key)`, `record_goal_anchor(sig)`, `to_dict()`, `from_dict(d)`; atributos `counts: dict[str,int]`, `_yield: dict[str,[float,int]]`, `goal_anchors: list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_novelty_model.py
import math
from agents.causal.perception import Scene, Object
from agents.causal.novelty import NoveltyModel, OPTIMISTIC_YIELD


def _scene(color=3, cell=(5, 5)):
    r, c = cell
    o = Object(color, frozenset([(r, c)]), (r, c, r, c), (float(r), float(c)), 1, "h")
    return Scene(objects=[o], grid=None)


def test_novelty_decreases_with_revisits():
    m = NoveltyModel()
    assert m.novelty("s") == 1.0                      # count 0 → 1/√1
    m.visit("s")
    assert math.isclose(m.novelty("s"), 1 / math.sqrt(2))
    m.visit("s")
    assert math.isclose(m.novelty("s"), 1 / math.sqrt(3))


def test_yield_estimate_optimistic_without_data():
    m = NoveltyModel()
    assert m.yield_estimate("ACTION6@cell=0,0") == OPTIMISTIC_YIELD


def test_observe_transition_updates_yield_and_counts():
    m = NoveltyModel()
    s = _scene()
    m.observe_transition("K", s)                      # 1º estado novo → novidade 1.0
    assert m.yield_estimate("K") == 1.0
    m.observe_transition("K", s)                       # revisita → novidade 1/√2
    assert math.isclose(m.yield_estimate("K"), (1.0 + 1 / math.sqrt(2)) / 2)


def test_record_goal_anchor_dedupes():
    m = NoveltyModel()
    m.record_goal_anchor("sigA")
    m.record_goal_anchor("sigA")
    m.record_goal_anchor("sigB")
    assert m.goal_anchors == ["sigA", "sigB"]


def test_roundtrip_serialization():
    m = NoveltyModel()
    m.observe_transition("K", _scene())
    m.record_goal_anchor("sigA")
    d = m.to_dict()
    m2 = NoveltyModel.from_dict(d)
    assert m2.to_dict() == d
    assert m2.yield_estimate("K") == m.yield_estimate("K")
    assert m2.goal_anchors == ["sigA"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v4 && uv run pytest tests/causal/test_novelty_model.py -q`
Expected: FAIL (ImportError: cannot import name `NoveltyModel`).

- [ ] **Step 3: Write minimal implementation**

Adicionar em `agents/causal/novelty.py`:

```python
class NoveltyModel:
    def __init__(self):
        self.counts = {}          # sig(str) -> int
        self._yield = {}          # action_key -> [soma, n]
        self.goal_anchors = []    # list[str]

    def count(self, sig) -> int:
        return self.counts.get(sig, 0)

    def novelty(self, sig) -> float:
        return 1.0 / math.sqrt(self.count(sig) + 1)

    def visit(self, sig) -> None:
        self.counts[sig] = self.counts.get(sig, 0) + 1

    def observe_transition(self, key, curr_scene) -> None:
        sig = state_signature(curr_scene)
        nov = self.novelty(sig)               # novidade ANTES de contar
        s, n = self._yield.get(key, [0.0, 0])
        self._yield[key] = [s + nov, n + 1]
        self.visit(sig)

    def yield_estimate(self, key) -> float:
        v = self._yield.get(key)
        if not v or v[1] == 0:
            return OPTIMISTIC_YIELD
        return v[0] / v[1]

    def record_goal_anchor(self, sig) -> None:
        if sig not in self.goal_anchors:
            self.goal_anchors.append(sig)

    def to_dict(self) -> dict:
        return {
            "counts": dict(self.counts),
            "yield": {k: list(v) for k, v in self._yield.items()},
            "goal_anchors": list(self.goal_anchors),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NoveltyModel":
        m = cls()
        m.counts = dict(d.get("counts", {}))
        m._yield = {k: list(v) for k, v in d.get("yield", {}).items()}
        m.goal_anchors = list(d.get("goal_anchors", []))
        return m
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v4 && uv run pytest tests/causal/test_novelty_model.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/novelty.py tests/causal/test_novelty_model.py
git commit -m "feat(causal): NoveltyModel (counts/yield/anchor/serialização)"
```

---

### Task 3: caminho v4 no `Policy.score`/`decide` (param opcional `novelty`)

**Files:**
- Modify: `agents/causal/policy.py` (`Policy.score`, `Policy.decide`)
- Test: `tests/causal/test_policy_novelty.py` (novo)

**Interfaces:**
- Consumes: `NoveltyModel.yield_estimate` (Task 2); `CausalModel.predict` → `(Effect|None, conf)`.
- Produces: `Policy.score(self, cand, model, seen_effects, budget_frac, novelty=None)` e `Policy.decide(self, scene, model, available_actions, seen_effects, budget_frac, novelty=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_policy_novelty.py
from arcengine import GameAction
from agents.causal.perception import Scene
from agents.causal.causal_model import CausalModel
from agents.causal.novelty import NoveltyModel
from agents.causal.policy import Policy, Candidate


def _c(key, has_object=False):
    return Candidate(GameAction.ACTION6, 5, 5, key, has_object)


def test_untried_key_scores_three_parity_with_v3():
    p = Policy(seed=0)
    model = CausalModel()
    nov = NoveltyModel()
    # chave inédita: y=1.0, ctrl=1.0 → 3.0 (igual ao termo v3 de ação nova)
    assert p.score(_c("K"), model, set(), 0.0, novelty=nov) == 3.0


def test_controllability_gate_prefers_reproducible():
    p = Policy(seed=0)
    nov = NoveltyModel()
    # chave A: efeito único reprodutível (conf 1.0); chave B: dois efeitos (conf 0.5)
    hi = CausalModel()
    hi.rules["A"] = {"moved:x": 4}
    lo = CausalModel()
    lo.rules["B"] = {"moved:x": 1, "recolored:y": 1}
    # yields otimistas iguais (1.0) → só o ctrl (conf) difere
    sa = p.score(_c("A"), hi, set(), 0.0, novelty=nov)
    sb = p.score(_c("B"), lo, set(), 0.0, novelty=nov)
    assert sa > sb


def test_none_key_penalty_applied():
    p = Policy(seed=0)
    nov = NoveltyModel()
    model = CausalModel()
    model.rules["K"] = {"none:None": 5}       # sempre none, conf 1.0
    # y otimista 1.0, ctrl=conf=1.0 → +3; −2 do none → 1.0. (Com uso real o yield
    # despenca por revisita e o score fica negativo; aqui validamos o termo −2.)
    assert p.score(_c("K"), model, set(), 0.0, novelty=nov) == 1.0


def test_novelty_none_reproduces_v3():
    p = Policy(seed=0)
    model = CausalModel()
    # sem novelty: chave inédita → +3 (caminho v3)
    assert p.score(_c("K"), model, set(), 0.0) == 3.0


def test_decide_accepts_novelty_kwarg():
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    nov = NoveltyModel()
    scene = Scene(objects=[], grid=None)
    cand = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0, novelty=nov)
    assert cand is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v4 && uv run pytest tests/causal/test_policy_novelty.py -q`
Expected: FAIL (`score`/`decide` não aceitam `novelty`).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/policy.py`, substituir `Policy.score` e a assinatura de `Policy.decide`:

```python
    def score(self, cand, model, seen_effects, budget_frac, novelty=None) -> float:
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
        return s

    def decide(self, scene, model, available_actions, seen_effects, budget_frac, novelty=None):
        cands = candidates(scene, available_actions)
        if not cands:
            return None
        if self._rng.random() < self.epsilon:
            return self._rng.choice(cands)
        best, best_s = None, None
        for c in cands:
            sc = self.score(c, model, seen_effects, budget_frac, novelty=novelty)
            if best_s is None or sc > best_s:
                best, best_s = c, sc
        return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v4 && uv run pytest tests/causal/test_policy_novelty.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/policy.py tests/causal/test_policy_novelty.py
git commit -m "feat(causal): caminho v4 no score/decide (novidade-de-estado + gate)"
```

---

### Task 4: wiring no `agent.py` + integração e regressão

**Files:**
- Modify: `agents/causal/agent.py`
- Test: `tests/causal/test_agent_novelty.py` (novo)

**Interfaces:**
- Consumes: `NoveltyModel`, `state_signature` (Tasks 1-2); `Policy.decide(..., novelty=)` (Task 3).
- Produces: `CausalObjectAgent` com `self._novelty: NoveltyModel`, atualizado no fecha-loop e passado ao `decide`; RESET não zera.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_agent_novelty.py
from arcengine import GameAction, GameState
from agents.causal.agent import CausalObjectAgent
from agents.causal.novelty import NoveltyModel


class _Frame:
    def __init__(self, frame, state=GameState.NOT_FINISHED, levels=0):
        self.frame = frame
        self.state = state
        self.levels_completed = levels
        self.available_actions = [GameAction.ACTION1]
        self.full_reset = False


def _grid(v):
    # pilha 1×8×8; célula (1,1) = v, resto 0 (fundo)
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = v
    return [g]


def test_agent_accumulates_novelty_over_steps():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    assert isinstance(a._novelty, NoveltyModel)
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))   # muda estado
    # após fechar o loop da 1ª ação, houve ao menos uma transição observada
    assert sum(v[1] for v in a._novelty._yield.values()) >= 1


def test_reset_does_not_wipe_novelty():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    a.choose_action([], _Frame(_grid(3)))
    a.choose_action([], _Frame(_grid(4)))
    a._novelty.counts["x"] = 7
    a.choose_action([], _Frame(_grid(3), state=GameState.GAME_OVER))  # RESET
    assert a._novelty.counts.get("x") == 7


def test_level_up_records_goal_anchor():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    a.choose_action([], _Frame(_grid(3), levels=0))
    a.choose_action([], _Frame(_grid(4), levels=1))   # level up no passo seguinte
    assert len(a._novelty.goal_anchors) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v4 && uv run pytest tests/causal/test_agent_novelty.py -q`
Expected: FAIL (`a._novelty` não existe).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/agent.py`:

(a) no topo, adicionar o import:

```python
from .novelty import NoveltyModel, state_signature
```

(b) em `_init_causal_state`, após `self._model = CausalModel()`:

```python
        self._novelty = NoveltyModel()
```

(c) no bloco de fecha-loop, logo após `self._seen_effects.add(actual.kind)`:

```python
            if level_up:
                self._novelty.record_goal_anchor(state_signature(self._prev_scene))
            self._novelty.observe_transition(self._last_key, scene)
```

(d) trocar a chamada do decide para passar a novidade:

```python
        cand = self._policy.decide(
            scene, self._model, latest_frame.available_actions or [GameAction.ACTION1],
            self._seen_effects, budget_frac, novelty=self._novelty,
        )
```

(e) adicionar ao `action.reasoning` a chave diagnóstica (dentro do dict `reasoning`):

```python
            "novelty_yield": round(self._novelty.yield_estimate(cand.key), 3),
```

(NÃO adicionar `self._novelty` ao branch de RESET — ele persiste.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v4 && uv run pytest tests/causal/test_agent_novelty.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Rodar a suíte inteira (regressão)**

Run: `cd .claude/worktrees/causal-v4 && uv run pytest tests/causal/ -q`
Expected: PASS (todos — 65 v1–v3 + novos v4).

- [ ] **Step 6: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_novelty.py
git commit -m "feat(causal): agent fia NoveltyModel (novidade-de-estado + âncora); regressão verde"
```

---

## Fora de escopo

- Perseguição explícita de meta a partir das âncoras (pós 1º level-up).
- Reuso de habilidades entre jogos (Passo 4).

## Validação ao vivo (pós-merge, fora do plano de código)

Rodar `CAUSAL_LOG=analysis/out/v4live/vc33.jsonl uv run main.py --agent=causalobject --game=vc33` (e `ls20`) e comparar vs v3: distribuição de chaves menos concentrada (menos fixação), e — meta real — chance de ≥1 level-up.
