# CausalObjectAgent v3 — Descoberta de clique (grade espacial) · Plano

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Endereçar e chavear cliques (`ACTION6`) por célula de uma grade espacial 6×6, para o modelo causal aprender interações por região e a policy varrer/explorar células que mexem no jogo.

**Architecture:** Mudança contida em `agents/causal/policy.py`: constante `GRID_N=6`, helpers célula↔ponto, `action_key` espacial, `candidates()` emitindo 36 candidatos de célula com flag `has_object`, e um termo de bônus no `score`. `causal_model.py`, `perception.py`, `hud.py`, `instrumentation.py`, `agent.py` **não mudam** (o `CausalModel` é agnóstico à string da chave).

**Tech Stack:** Python 3.12, numpy/stdlib puro, pytest, `arcengine.GameAction`. Sem LLM/GPU. Kaggle-submittable.

## Global Constraints

- Numpy/stdlib puro; nenhuma dependência nova; nada de LLM/GPU.
- `GRID_N = 6` (grade 6×6 = 36 células) sobre a grade fixa 64×64.
- Convenção de eixos: `x` = coluna, `y` = linha; `Object.cells` e `Object.centroid` são `(row, col)`.
- Chave de clique: `f"{action.name}@cell={gx},{gy}"`; ação simples: `action.name`.
- Não alterar `causal_model.py`, `perception.py`, `hud.py`, `instrumentation.py`, `agent.py`.
- Os 50 testes v1+v2 em `tests/causal/` devem seguir verdes.

---

### Task 1: Grade espacial (helpers célula↔ponto)

**Files:**
- Modify: `agents/causal/policy.py` (topo: constante `GRID_N` + funções `cell_center`, `cell_of`)
- Test: `tests/causal/test_policy_grid.py` (novo)

**Interfaces:**
- Consumes: nada.
- Produces:
  - `GRID_N: int = 6`
  - `cell_center(gx: int, gy: int) -> tuple[int, int]` → `(x, y)` centro da célula, ambos em `0..63`.
  - `cell_of(x: int, y: int) -> tuple[int, int]` → `(gx, gy)` índice de célula, ambos em `0..GRID_N-1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_policy_grid.py
from agents.causal.policy import GRID_N, cell_center, cell_of


def test_grid_n_is_six():
    assert GRID_N == 6


def test_cell_center_in_bounds_and_distinct():
    seen = set()
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            x, y = cell_center(gx, gy)
            assert 0 <= x <= 63 and 0 <= y <= 63
            assert (x, y) not in seen
            seen.add((x, y))
    assert len(seen) == GRID_N * GRID_N


def test_cell_centers_expected_values():
    # (gx+0.5)*64/6 truncado: {5,16,26,37,48,58}
    xs = sorted({cell_center(gx, 0)[0] for gx in range(GRID_N)})
    assert xs == [5, 16, 26, 37, 48, 58]


def test_cell_of_roundtrips_center():
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            x, y = cell_center(gx, gy)
            assert cell_of(x, y) == (gx, gy)


def test_cell_of_clamps_edges():
    assert cell_of(0, 0) == (0, 0)
    assert cell_of(63, 63) == (GRID_N - 1, GRID_N - 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v3 && uv run pytest tests/causal/test_policy_grid.py -q`
Expected: FAIL (ImportError: cannot import name `GRID_N` / `cell_center`).

- [ ] **Step 3: Write minimal implementation**

Adicionar ao topo de `agents/causal/policy.py`, logo após os imports e antes de `Candidate`:

```python
GRID_N = 6


def cell_center(gx: int, gy: int) -> tuple[int, int]:
    x = int((gx + 0.5) * 64 / GRID_N)
    y = int((gy + 0.5) * 64 / GRID_N)
    return x, y


def cell_of(x: int, y: int) -> tuple[int, int]:
    gx = min(GRID_N - 1, int(x) * GRID_N // 64)
    gy = min(GRID_N - 1, int(y) * GRID_N // 64)
    return gx, gy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v3 && uv run pytest tests/causal/test_policy_grid.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/policy.py tests/causal/test_policy_grid.py
git commit -m "feat(causal): grade espacial 6×6 (cell_center/cell_of)"
```

---

### Task 2: Chave de ação espacial + campo `has_object` no Candidate

**Files:**
- Modify: `agents/causal/policy.py` (`Candidate` namedtuple; `action_key`)
- Test: `tests/causal/test_policy_key.py` (novo)

**Interfaces:**
- Consumes: `GRID_N` (Task 1).
- Produces:
  - `Candidate = namedtuple("Candidate", "action x y key has_object")` (5 campos).
  - `action_key(action, cell=None) -> str`: simples → `action.name`; complexa com `cell=(gx,gy)` → `f"{action.name}@cell={gx},{gy}"`; complexa com `cell=None` → `f"{action.name}@empty"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_policy_key.py
from arcengine import GameAction
from agents.causal.policy import Candidate, action_key


def test_candidate_has_five_fields():
    c = Candidate(GameAction.ACTION1, None, None, "ACTION1", False)
    assert c.has_object is False
    assert c._fields == ("action", "x", "y", "key", "has_object")


def test_action_key_simple_is_name():
    assert action_key(GameAction.ACTION1) == "ACTION1"


def test_action_key_complex_same_cell_same_key():
    a = GameAction.ACTION6
    assert action_key(a, (2, 3)) == "ACTION6@cell=2,3"
    assert action_key(a, (2, 3)) == action_key(a, (2, 3))


def test_action_key_complex_diff_cell_diff_key():
    a = GameAction.ACTION6
    assert action_key(a, (2, 3)) != action_key(a, (3, 2))


def test_action_key_complex_none_cell_is_empty():
    assert action_key(GameAction.ACTION6, None) == "ACTION6@empty"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v3 && uv run pytest tests/causal/test_policy_key.py -q`
Expected: FAIL (Candidate tem 4 campos; `action_key` assinatura antiga usa `target_obj.color`).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/policy.py`, trocar a definição do namedtuple:

```python
Candidate = namedtuple("Candidate", "action x y key has_object")
```

e substituir `action_key` inteira por:

```python
def action_key(action, cell=None) -> str:
    if not action.is_complex():
        return action.name
    if cell is None:
        return f"{action.name}@empty"
    gx, gy = cell
    return f"{action.name}@cell={gx},{gy}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v3 && uv run pytest tests/causal/test_policy_key.py -q`
Expected: PASS (5 passed). Nota: `test_policy_candidates.py` e `test_policy_decide.py` podem quebrar aqui por causa da mudança de `candidates()`/`Candidate`; serão consertados na Task 3/4. Rodar só o arquivo novo neste passo.

- [ ] **Step 5: Commit**

```bash
git add agents/causal/policy.py tests/causal/test_policy_key.py
git commit -m "feat(causal): action_key espacial por célula + Candidate.has_object"
```

---

### Task 3: `candidates()` emite 36 candidatos de célula com `has_object`

**Files:**
- Modify: `agents/causal/policy.py` (`candidates`; novo helper `_object_cells`)
- Test: `tests/causal/test_policy_candidates.py` (reescrever para a nova semântica)

**Interfaces:**
- Consumes: `GRID_N`, `cell_center`, `cell_of`, `action_key`, `Candidate` (Tasks 1-2); `Scene`/`Object` de `agents/causal/perception.py` (`scene.objects`, `Object.cells` em `(row,col)`).
- Produces:
  - `_object_cells(scene) -> set[tuple[int,int]]`: conjunto de `(gx,gy)` tocados por algum objeto de primeiro plano.
  - `candidates(scene, available_actions) -> list[Candidate]`: simples → 1 candidato; complexa → `GRID_N*GRID_N` candidatos (um por célula), com `x,y` = centro e `has_object` correto.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_policy_candidates.py  (substituir conteúdo)
from arcengine import GameAction
from agents.causal.perception import Scene, Object
from agents.causal.policy import candidates, cell_of, GRID_N, _object_cells


def _obj(cells, color=3):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return Object(color, cset, bbox, centroid, len(cells), "h")


def test_simple_action_one_candidate():
    scene = Scene(objects=[], grid=None)
    out = candidates(scene, [GameAction.ACTION1])
    assert len(out) == 1
    assert out[0].key == "ACTION1"
    assert out[0].x is None and out[0].y is None


def test_complex_action_emits_grid_candidates():
    scene = Scene(objects=[], grid=None)
    out = candidates(scene, [GameAction.ACTION6])
    assert len(out) == GRID_N * GRID_N
    keys = {c.key for c in out}
    assert len(keys) == GRID_N * GRID_N          # chaves distintas
    pts = {(c.x, c.y) for c in out}
    assert len(pts) == GRID_N * GRID_N           # pontos distintos
    for c in out:
        assert 0 <= c.x <= 63 and 0 <= c.y <= 63


def test_object_cells_marks_occupied_cell():
    # objeto em (row=5,col=5) → célula gx=0,gy=0
    scene = Scene(objects=[_obj([(5, 5)])], grid=None)
    occ = _object_cells(scene)
    assert occ == {(0, 0)}


def test_has_object_flag_only_on_occupied_cell():
    scene = Scene(objects=[_obj([(5, 5)])], grid=None)
    out = candidates(scene, [GameAction.ACTION6])
    occupied = [c for c in out if c.has_object]
    assert len(occupied) == 1
    assert cell_of(occupied[0].x, occupied[0].y) == (0, 0)


def test_empty_scene_no_has_object():
    scene = Scene(objects=[], grid=None)
    out = candidates(scene, [GameAction.ACTION6])
    assert all(c.has_object is False for c in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v3 && uv run pytest tests/causal/test_policy_candidates.py -q`
Expected: FAIL (`_object_cells` inexistente; `candidates` ainda gera por-objeto).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/policy.py`, substituir `candidates` e adicionar `_object_cells`:

```python
def _object_cells(scene) -> set:
    occ = set()
    for o in scene.objects:
        for (r, c) in o.cells:
            occ.add(cell_of(c, r))   # x=col, y=row
    return occ


def candidates(scene, available_actions) -> list:
    out = []
    occ = _object_cells(scene)
    for a in available_actions:
        action = _as_action(a)
        if not action.is_complex():
            out.append(Candidate(action, None, None, action.name, False))
        else:
            for gy in range(GRID_N):
                for gx in range(GRID_N):
                    x, y = cell_center(gx, gy)
                    out.append(
                        Candidate(action, x, y, action_key(action, (gx, gy)),
                                  (gx, gy) in occ)
                    )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v3 && uv run pytest tests/causal/test_policy_candidates.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/policy.py tests/causal/test_policy_candidates.py
git commit -m "feat(causal): candidates() emite 36 células com has_object"
```

---

### Task 4: Bônus de presença de objeto no score + integração e regressão

**Files:**
- Modify: `agents/causal/policy.py` (`Policy.score`)
- Test: `tests/causal/test_policy_decide.py` (ajustar/estender), `tests/causal/test_policy_sweep.py` (novo)

**Interfaces:**
- Consumes: `candidates`, `Candidate.has_object`, `action_key`, `cell_of` (Tasks 1-3); `CausalModel` (`observe(prev, key, curr)`, `predict(key)`, `is_progress(key)`).
- Produces: `Policy.score(cand, model, seen_effects, budget_frac)` soma `+0.5` quando `cand.has_object`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_policy_sweep.py
from arcengine import GameAction
from agents.causal.perception import Scene, Object
from agents.causal.causal_model import CausalModel
from agents.causal.policy import Policy, candidates, cell_of


def _obj(cells, color=3):
    cset = frozenset(cells)
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))
    centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
    return Object(color, cset, bbox, centroid, len(cells), "h")


def test_object_cell_preferred_over_empty_when_unexplored():
    # objeto em (row=50,col=50) → célula (4,4), NÃO a primeira da ordem de
    # geração (0,0). Sem o bônus, o empate entre células inexploradas cairia
    # em (0,0); com o bônus +0.5, a policy escolhe a célula ocupada (4,4).
    scene = Scene(objects=[_obj([(50, 50)])], grid=None)
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    chosen = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0)
    assert cell_of(chosen.x, chosen.y) == (4, 4)          # bônus +0.5 vence


def test_sweep_avoids_known_none_cell():
    scene = Scene(objects=[], grid=None)
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    first = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0)
    # registra a célula escolhida como 'none' (empty→empty gera Effect none)
    empty = Scene(objects=[], grid=None)
    model.observe(empty, first.key, empty)
    second = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0)
    assert second.key != first.key                        # não repete célula morta


def test_reproducible_cell_key_is_stable_and_predictable():
    scene = Scene(objects=[], grid=None)
    p = Policy(seed=0, epsilon=0.0)
    model = CausalModel()
    c = p.decide(scene, model, [GameAction.ACTION6], set(), 0.0)
    # cena→cena seguinte com mudança real sob a mesma chave de célula
    before = Scene(objects=[_obj([(5, 5)])], grid=None)
    after = Scene(objects=[_obj([(6, 6)], color=4)], grid=None)
    eff = model.observe(before, c.key, after)
    assert eff.kind != "none"
    pred, conf = model.predict(c.key)
    assert pred is not None and pred.kind == eff.kind
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/causal-v3 && uv run pytest tests/causal/test_policy_sweep.py -q`
Expected: FAIL em `test_object_cell_preferred_over_empty_when_unexplored` (sem o bônus, todas as células inexploradas empatam e a ordem determinística pode não escolher a ocupada).

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/policy.py`, dentro de `Policy.score`, adicionar o termo de bônus (logo antes do `return s`):

```python
        if cand.has_object:
            s += 0.5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd .claude/worktrees/causal-v3 && uv run pytest tests/causal/test_policy_sweep.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Ajustar `test_policy_decide.py` e rodar a suíte inteira**

`tests/causal/test_policy_decide.py` foi escrito para a semântica antiga (candidatos por-objeto, `Candidate` de 4 campos). Abrir o arquivo; para cada construção manual de `Candidate(...)`, acrescentar o 5º argumento `False` (ou `has_object=False`); trocar qualquer expectativa que dependa de `action_key(action, obj)` por chave de célula. Não enfraquecer asserts — se um teste afirmava "prefere ação não-tentada sobre nada-sabido", mantê-lo com candidatos de célula. Então:

Run: `cd .claude/worktrees/causal-v3 && uv run pytest tests/causal/ -q`
Expected: PASS (todos — 50 v1+v2 já existentes, atualizados onde a assinatura mudou, + os novos de v3).

- [ ] **Step 6: Commit**

```bash
git add agents/causal/policy.py tests/causal/test_policy_sweep.py tests/causal/test_policy_decide.py
git commit -m "feat(causal): bônus has_object no score; varredura dirigida + regressão verde"
```

---

## Fora de escopo

- Modelo de objetivo / sinal de progresso (Passo 3).
- Refino coarse-to-fine da grade; chave relacional entre objetos; reuso entre jogos.

## Validação ao vivo (pós-merge, fora do plano de código)

Rodar `CAUSAL_LOG=analysis/out/v3live/vc33.jsonl uv run main.py --agent=causalobject --game=vc33` e conferir: chaves de clique distintas ≤ 36 (era ~77) e ao menos uma chave `@cell=` com efeito não-`none` repetida.
