# CausalObjectAgent v2 Implementation Plan — HUD-masking + IoU matching

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Limpar o sinal do efeito do CausalObjectAgent — mascarar o HUD (regiões que mudam a cada ação) e tornar a identidade de objeto tolerante a mudança de forma (IoU) — para o modelo causal aprender dinâmica real em vez de ruído.

**Architecture:** Adiciona `HudMask` (contador online de mudança por célula), faz `parse` excluir células mascaradas, troca o matching estrito por 3 tiers com IoU, adiciona o efeito `morphed`, e fia tudo no `choose_action`. Não toca a Policy.

**Tech Stack:** Python 3.12, numpy, pytest. Base v1 em `agents/causal/` (worktree `causal-v2`, já tem o v1 merged, 36 testes verdes).

## Global Constraints

- **Só numpy + stdlib.** Sem GPU/LLM/scipy/torch/internet em runtime (Kaggle-clean).
- **Não alterar a Policy** (`policy.py`) nem a Instrumentation. v2 = percepção/efeito só.
- **Manter os 36 testes v1 verdes** — `parse(hud_mask=None)` e `match_objects` devem preservar o comportamento v1 quando não há máscara/mudança de forma.
- **Thresholds (verbatim do spec):** `HUD_THRESHOLD = 0.7`, `HUD_MIN_SAMPLES = 5`, `IOU_THRESHOLD = 0.3`.
- Grade 64×64 ints 0–15; `frame` pode ser pilha (usar última camada).
- Rodar testes: `cd <worktree> && uv run pytest tests/causal/ -v`.
- `Effect` kinds passam a incluir `morphed`: {none, moved, appeared, disappeared, recolored, morphed, structural}.

---

### Task 1: `HudMask` (novo módulo)

**Files:**
- Create: `agents/causal/hud.py`
- Test: `tests/causal/test_hud.py`

**Interfaces:**
- Consumes: numpy; grades 2D `np.ndarray` (64×64 ints).
- Produces: `class HudMask` com `update(prev_grid, curr_grid) -> None`, `mask() -> np.ndarray(bool)`, `to_dict()/from_dict(d)`. Constantes de módulo `HUD_THRESHOLD=0.7`, `HUD_MIN_SAMPLES=5`.

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_hud.py
import numpy as np
from agents.causal.hud import HudMask


def _pair(changed_cells, shape=(8, 8)):
    a = np.zeros(shape, dtype=int)
    b = np.zeros(shape, dtype=int)
    for (r, c) in changed_cells:
        b[r, c] = 5
    return a, b


def test_mask_empty_before_min_samples():
    h = HudMask()
    for _ in range(4):                       # < HUD_MIN_SAMPLES (5)
        a, b = _pair([(0, 0)])
        h.update(a, b)
    assert not h.mask().any()                # cego ainda


def test_cell_changing_every_step_is_masked():
    h = HudMask()
    for _ in range(6):                       # >= 5 amostras, muda toda vez
        a, b = _pair([(0, 0)])
        h.update(a, b)
    m = h.mask()
    assert m[0, 0]                           # HUD
    assert not m[3, 3]                       # nunca mudou


def test_cell_changing_once_not_masked():
    h = HudMask()
    a, b = _pair([(1, 1)]); h.update(a, b)   # muda 1x
    for _ in range(9):                       # 9 transições sem mudanca
        z = np.zeros((8, 8), dtype=int)
        h.update(z, z)
    assert not h.mask()[1, 1]                # 1/10 = 0.1 < 0.7


def test_serialization_roundtrip():
    h = HudMask()
    for _ in range(6):
        a, b = _pair([(0, 1)]); h.update(a, b)
    h2 = HudMask.from_dict(h.to_dict())
    assert np.array_equal(h2.mask(), h.mask())
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd <worktree> && uv run pytest tests/causal/test_hud.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implementar `hud.py`**

```python
# agents/causal/hud.py
from __future__ import annotations

import numpy as np

HUD_THRESHOLD = 0.7
HUD_MIN_SAMPLES = 5


class HudMask:
    def __init__(self, shape=(64, 64)):
        self._shape = tuple(shape)
        self.change_count = np.zeros(self._shape, dtype=int)
        self.total = 0

    def update(self, prev_grid, curr_grid) -> None:
        a = np.asarray(prev_grid); b = np.asarray(curr_grid)
        if a.shape != self.change_count.shape:
            # (re)inicializa se a grade tem outro tamanho
            self._shape = a.shape
            self.change_count = np.zeros(self._shape, dtype=int)
            self.total = 0
        self.change_count += (a != b).astype(int)
        self.total += 1

    def mask(self) -> np.ndarray:
        if self.total < HUD_MIN_SAMPLES:
            return np.zeros(self._shape, dtype=bool)
        return (self.change_count / self.total) >= HUD_THRESHOLD

    def to_dict(self) -> dict:
        return {"shape": list(self._shape),
                "change_count": self.change_count.tolist(),
                "total": self.total}

    @classmethod
    def from_dict(cls, d: dict) -> "HudMask":
        h = cls(tuple(d["shape"]))
        h.change_count = np.array(d["change_count"], dtype=int)
        h.total = d["total"]
        return h
```

- [ ] **Step 4: Rodar e ver passar**

Run: `cd <worktree> && uv run pytest tests/causal/test_hud.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/hud.py tests/causal/test_hud.py
git commit -m "feat(causal): HudMask — deteccao de HUD por frequencia de mudanca por celula"
```

---

### Task 2: `parse(frame, hud_mask=None)` + `to_grid` público

**Files:**
- Modify: `agents/causal/perception.py`
- Test: `tests/causal/test_perception_hud.py`

**Interfaces:**
- Consumes: `parse`, `_to_grid`, `_background_color` (já existem).
- Produces:
  - `to_grid(frame) -> np.ndarray` — alias público de `_to_grid` (o agente usará para alimentar o `HudMask`).
  - `parse(frame, hud_mask=None) -> Scene` — quando `hud_mask` (bool 64×64) é dado, células mascaradas são tratadas como fundo (excluídas da segmentação). `hud_mask=None` = comportamento v1 idêntico.

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_perception_hud.py
import numpy as np
from agents.causal.perception import parse, to_grid


def test_to_grid_reduces_stack():
    layer = np.zeros((3, 3), dtype=int)
    layer[0, 0] = 9
    g = to_grid([np.zeros((3, 3), dtype=int).tolist(), layer.tolist()])
    assert g[0, 0] == 9 and g.shape == (3, 3)


def test_hud_mask_excludes_masked_object():
    grid = np.zeros((6, 6), dtype=int)
    grid[0, 0] = 3            # objeto no HUD
    grid[4, 4] = 7            # objeto de gameplay
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True         # mascara o (0,0)
    scene = parse(grid.tolist(), hud_mask=mask)
    colors = sorted(o.color for o in scene.objects)
    assert colors == [7]      # o objeto mascarado sumiu da Scene


def test_no_mask_is_v1_behavior():
    grid = np.zeros((6, 6), dtype=int)
    grid[0, 0] = 3
    grid[4, 4] = 7
    scene = parse(grid.tolist())               # sem mask
    assert len(scene.objects) == 2
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd <worktree> && uv run pytest tests/causal/test_perception_hud.py -v`
Expected: FAIL (`cannot import name 'to_grid'` / TypeError hud_mask).

- [ ] **Step 3: Modificar `perception.py`**

Adicionar o alias público logo após `_to_grid`:

```python
def to_grid(frame) -> np.ndarray:
    return _to_grid(frame)
```

Alterar a assinatura e o loop de `parse` (adicionar o parâmetro e a checagem de máscara):

```python
def parse(frame, hud_mask=None) -> Scene:
    grid = _to_grid(frame)
    bg = _background_color(grid)
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            masked = hud_mask is not None and hud_mask[r, c]
            if seen[r, c] or grid[r, c] == bg or masked:
                continue
            color = int(grid[r, c])
            cells = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    nmasked = hud_mask is not None and hud_mask[nr, nc]
                    if not seen[nr, nc] and grid[nr, nc] == color and not nmasked:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            cset = frozenset(cells)
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            centroid = (sum(rs) / len(rs), sum(cs) / len(cs))
            objects.append(
                Object(color, cset, bbox, centroid, len(cells), _shape_hash(cset))
            )
    return Scene(objects=objects, grid=grid)
```

Nota: o guard de limites `if not (0 <= nr < rows and 0 <= nc < cols): continue` vem ANTES de acessar `hud_mask[nr, nc]`, evitando índice fora do intervalo.

- [ ] **Step 4: Rodar e ver passar (incl. regressão v1)**

Run: `cd <worktree> && uv run pytest tests/causal/test_perception_hud.py tests/causal/test_perception.py -v`
Expected: PASS (3 novos + 4 v1 de `test_perception.py`).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/perception.py tests/causal/test_perception_hud.py
git commit -m "feat(causal): parse(hud_mask=) exclui HUD + to_grid publico"
```

---

### Task 3: `match_objects` por IoU (3 tiers)

**Files:**
- Modify: `agents/causal/perception.py`
- Test: `tests/causal/test_perception_iou.py`

**Interfaces:**
- Consumes: `Object`, `Scene`, `replace`, `_id_counter`.
- Produces:
  - `_iou(cells_a, cells_b) -> float` — interseção/união de dois `frozenset` de células.
  - `match_objects(prev, curr) -> Scene` reescrito com 3 tiers: (1) exato cor+shape_hash+centroide; (2) IoU≥`IOU_THRESHOLD` mesma cor; (3) IoU≥`IOU_THRESHOLD` qualquer cor. `IOU_THRESHOLD=0.3` como constante de módulo. Um `prev` casa no máximo um `curr` (`used`). Sem casamento → id fresco. Preserva o comportamento v1 nos testes existentes (`test_perception_match.py`).

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_perception_iou.py
import numpy as np
from agents.causal.perception import parse, match_objects


def test_growing_object_keeps_id():
    g0 = np.zeros((8, 8), dtype=int); g0[2, 2] = 3; g0[2, 3] = 3   # 2 células
    s0 = match_objects(None, parse(g0.tolist()))
    id0 = s0.objects[0].id
    g1 = np.zeros((8, 8), dtype=int); g1[2, 2] = 3; g1[2, 3] = 3; g1[2, 4] = 3  # cresce p/ 3
    s1 = match_objects(s0, parse(g1.tolist()))
    assert len(s1.objects) == 1
    assert s1.objects[0].id == id0            # IoU = 2/3 ≥ 0.3 → mantém id


def test_distinct_distant_objects_do_not_merge():
    g0 = np.zeros((10, 10), dtype=int); g0[1, 1] = 3
    s0 = match_objects(None, parse(g0.tolist()))
    g1 = np.zeros((10, 10), dtype=int); g1[1, 1] = 3; g1[8, 8] = 7   # novo objeto longe
    s1 = match_objects(s0, parse(g1.tolist()))
    ids = sorted(o.id for o in s1.objects)
    assert len(set(ids)) == 2                  # não funde


def test_recolor_plus_reshape_keeps_id_via_tier3():
    g0 = np.zeros((8, 8), dtype=int); g0[2, 2] = 3; g0[2, 3] = 3
    s0 = match_objects(None, parse(g0.tolist()))
    id0 = s0.objects[0].id
    g1 = np.zeros((8, 8), dtype=int); g1[2, 2] = 7; g1[2, 3] = 7; g1[2, 4] = 7  # cor nova + cresce
    s1 = match_objects(s0, parse(g1.tolist()))
    assert s1.objects[0].id == id0             # tier 3 (IoU qualquer cor)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd <worktree> && uv run pytest tests/causal/test_perception_iou.py -v`
Expected: FAIL (id muda / objetos não casam sob mudança de forma).

- [ ] **Step 3: Reescrever `match_objects` + `_iou`**

Adicionar a constante no topo do módulo (junto de `_id_counter`):

```python
IOU_THRESHOLD = 0.3
```

Adicionar o helper e substituir `match_objects` inteiro por:

```python
def _iou(a, b) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def match_objects(prev, curr) -> Scene:
    prev_objs = list(prev.objects) if prev is not None else []
    used = set()
    matched = {}  # indice em curr.objects -> id do prev

    # tier 1: exato cor + shape_hash + centroide mais proximo
    for i, o in enumerate(curr.objects):
        best, best_d = None, None
        for p in prev_objs:
            if p.id in used or p.shape_hash != o.shape_hash or p.color != o.color:
                continue
            d = (p.centroid[0] - o.centroid[0]) ** 2 + (p.centroid[1] - o.centroid[1]) ** 2
            if best_d is None or d < best_d:
                best, best_d = p, d
        if best is not None:
            used.add(best.id); matched[i] = best.id

    # tier 2: IoU mesma cor  /  tier 3: IoU qualquer cor
    for require_color in (True, False):
        for i, o in enumerate(curr.objects):
            if i in matched:
                continue
            best, best_iou = None, None
            for p in prev_objs:
                if p.id in used:
                    continue
                if require_color and p.color != o.color:
                    continue
                iou = _iou(p.cells, o.cells)
                if iou >= IOU_THRESHOLD and (best_iou is None or iou > best_iou):
                    best, best_iou = p, iou
            if best is not None:
                used.add(best.id); matched[i] = best.id

    new_objs = []
    for i, o in enumerate(curr.objects):
        oid = matched.get(i)
        if oid is None:
            oid = next(_id_counter)
        new_objs.append(replace(o, id=oid))
    return Scene(objects=new_objs, grid=curr.grid)
```

Nota: tier 1 agora exige cor igual (era o passe-1 v1). O antigo passe-2 v1 (fallback color-agnostic por distância ≤2.0, que habilitava `recolored`) é substituído pelos tiers IoU 2/3, que cobrem o mesmo caso (recolor in-place tem IoU=1.0 ≥ 0.3) e mais (mudança de forma). Rode `test_perception_match.py` para confirmar que os casos v1 (mesmo id após mover; objeto novo ganha id) seguem verdes.

- [ ] **Step 4: Rodar e ver passar (incl. regressão v1)**

Run: `cd <worktree> && uv run pytest tests/causal/test_perception_iou.py tests/causal/test_perception_match.py tests/causal/test_effect.py -v`
Expected: PASS (3 novos + os de match + os de effect v1, incl. `test_recolored_object_keeps_id`).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/perception.py tests/causal/test_perception_iou.py
git commit -m "feat(causal): match_objects por IoU (3 tiers) tolerante a mudanca de forma"
```

---

### Task 4: `compute_effect` — efeito `morphed`

**Files:**
- Modify: `agents/causal/causal_model.py`
- Test: `tests/causal/test_effect_morphed.py`

**Interfaces:**
- Produces: `compute_effect(prev, curr) -> Effect` com o kind `morphed` para: mesmo id, centroide não moveu (`dr=dc=0`), cor igual, mas `cells` diferentes (tamanho/forma mudou). Ordem por objeto: `moved` → `recolored` → `morphed`. `moved`/`recolored`/`none`/`structural`/`appeared`/`disappeared` inalterados.

- [ ] **Step 1: Escrever os testes (falham)**

```python
# tests/causal/test_effect_morphed.py
import numpy as np
from agents.causal.perception import parse, match_objects
from agents.causal.causal_model import compute_effect


def _scene(prev, grid):
    return match_objects(prev, parse(grid))


def test_morphed_on_inplace_resize():
    # objeto simétrico em cruz cujo centroide NÃO muda ao ganhar uma célula simétrica
    g0 = np.zeros((9, 9), dtype=int)
    g0[4, 4] = 3; g0[4, 3] = 3; g0[4, 5] = 3          # linha horizontal, centroide (4,4)
    s0 = _scene(None, g0.tolist())
    g1 = np.zeros((9, 9), dtype=int)
    g1[4, 4] = 3; g1[4, 3] = 3; g1[4, 5] = 3; g1[3, 4] = 3; g1[5, 4] = 3  # cruz, centroide (4,4)
    s1 = _scene(s0, g1.tolist())
    e = compute_effect(s0, s1)
    assert e.kind == "morphed"                        # id mantido (IoU 3/5), centroide igual, cells mudou


def test_moved_still_moved():
    g0 = np.zeros((8, 8), dtype=int); g0[1, 1] = 3
    s0 = _scene(None, g0.tolist())
    g1 = np.zeros((8, 8), dtype=int); g1[1, 4] = 3
    s1 = _scene(s0, g1.tolist())
    assert compute_effect(s0, s1).kind == "moved"
```

Nota p/ o implementer: o exemplo de `test_morphed_on_inplace_resize` foi escolhido para manter o centroide em (4,4) antes e depois (cruz simétrica), garantindo `dr=dc=0` e `cells` diferentes → `morphed`. Se ao rodar der `moved`, verifique o centroide computado e ajuste as células para simetria antes de fixar.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd <worktree> && uv run pytest tests/causal/test_effect_morphed.py -v`
Expected: FAIL (efeito volta `none` ou `structural`, não `morphed`).

- [ ] **Step 3: Modificar `compute_effect`**

No branch de mesmo-id, adicionar o ramo `morphed` após `recolored`:

```python
        else:
            dr = round(co.centroid[0] - po.centroid[0])
            dc = round(co.centroid[1] - po.centroid[1])
            if (dr, dc) != (0, 0):
                changes.append(Effect("moved", (dr, dc)))
            elif co.color != po.color:
                changes.append(Effect("recolored", (po.color, co.color)))
            elif co.cells != po.cells:
                changes.append(Effect("morphed", (po.size, co.size)))
```

- [ ] **Step 4: Rodar e ver passar (incl. regressão)**

Run: `cd <worktree> && uv run pytest tests/causal/test_effect_morphed.py tests/causal/test_effect.py tests/causal/test_causal_model.py -v`
Expected: PASS (2 novos + effect v1 + causal_model v1).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/causal_model.py tests/causal/test_effect_morphed.py
git commit -m "feat(causal): compute_effect kind 'morphed' (redimensiona in-place mantendo id)"
```

---

### Task 5: Wiring do `HudMask` no `CausalObjectAgent` + regressão HUD

**Files:**
- Modify: `agents/causal/agent.py`
- Test: `tests/causal/test_agent_hud.py`

**Interfaces:**
- Consumes: `HudMask` (Task 1), `to_grid`/`parse(hud_mask=)` (Task 2).
- Produces: `choose_action` que, a cada passo, atualiza o `HudMask` com `(prev_grid, grid)` e passa `self._hud.mask()` ao `parse`. Novo estado `self._hud` e `self._prev_grid` em `_init_causal_state`, limpos no RESET.

- [ ] **Step 1: Escrever o teste de regressão HUD (falha)**

```python
# tests/causal/test_agent_hud.py
import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.causal.agent import CausalObjectAgent


def _agent():
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.frames = []
    a.action_counter = 0
    a._init_causal_state()
    return a


def _frame(grid):
    return FrameData(levels_completed=0, state=GameState.NOT_FINISHED,
                     frame=[grid], available_actions=[GameAction.ACTION6.value])


def _hud_grid(tick):
    # gameplay estático embaixo; um "contador" no topo (linha 0) que muda a cada passo
    g = np.zeros((10, 10), dtype=int)
    g[5, 5] = 3                 # objeto de gameplay ESTÁTICO
    g[0, tick % 10] = 4         # HUD: célula que muda de posição a cada passo
    return g.tolist()


def test_hud_only_change_becomes_none_after_learning():
    a = _agent()
    # alimenta várias transições onde SÓ o HUD (linha 0) muda
    for t in range(8):
        a.choose_action([], _frame(_hud_grid(t)))
    # após aprender o HUD, os efeitos observados recentes devem virar 'none'
    # (o gameplay em (5,5) nunca mudou; só a linha 0, agora mascarada)
    recent = [r["actual"] for r in a._instr.records if r["actual"] is not None]
    assert recent and recent[-1] == "none"


def test_hud_mask_is_reset_on_reset():
    a = _agent()
    for t in range(6):
        a.choose_action([], _frame(_hud_grid(t)))
    assert a._hud.total >= 5
    over = FrameData(levels_completed=0, state=GameState.GAME_OVER, frame=[_hud_grid(0)])
    a.choose_action([], over)
    assert a._hud.total == 0            # resetado
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd <worktree> && uv run pytest tests/causal/test_agent_hud.py -v`
Expected: FAIL (AttributeError `_hud` / efeito não vira none).

- [ ] **Step 3: Modificar `agent.py`**

Adicionar aos imports do topo:

```python
from .perception import parse, match_objects, to_grid
from .hud import HudMask
```

Em `_init_causal_state`, adicionar (junto das outras inicializações):

```python
        self._hud = HudMask()
        self._prev_grid = None
```

No RESET (dentro do `if` de NOT_PLAYED/GAME_OVER/full_reset, junto das limpezas existentes), adicionar:

```python
            self._hud = HudMask()
            self._prev_grid = None
```

Substituir a linha que calcula `scene`:

```python
        scene = match_objects(self._prev_scene, parse(latest_frame.frame))
```

por o bloco que atualiza o HUD e mascara:

```python
        grid = to_grid(latest_frame.frame)
        if self._prev_grid is not None:
            self._hud.update(self._prev_grid, grid)
        scene = match_objects(self._prev_scene, parse(latest_frame.frame, hud_mask=self._hud.mask()))
```

E no final do passo, junto de `self._prev_scene = scene`, adicionar:

```python
        self._prev_grid = grid
```

(mantendo tudo o mais do `choose_action` idêntico — Policy, logging deferido, etc.)

- [ ] **Step 4: Rodar toda a suíte causal**

Run: `cd <worktree> && uv run pytest tests/causal/ -v`
Expected: PASS — os 36 testes v1 + todos os novos das Tasks 1–5.

- [ ] **Step 5: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_hud.py
git commit -m "feat(causal): fia HudMask no choose_action (mascara HUD antes do efeito)"
```

---

## Notas de execução

- **Ordem:** Tasks 1–4 são independentes entre si (HudMask, parse, match, effect); a Task 5 integra tudo. Fazer em sequência 1→5.
- **Regressão v1 é obrigatória:** cada task roda também os testes v1 relevantes; a Task 5 roda a suíte inteira. Se algum dos 36 v1 quebrar, é um defeito da task.
- **Fixture HUD sintético (não recording real):** o spec §5 menciona um fixture com o par de frames real do vc33, mas o recording é gitignored (fora do worktree). Usamos um fixture sintético equivalente (linha 0 muda a cada passo, gameplay estático) — commitável e determinístico. Desvio deliberado registrado aqui.
- **Validação pós-merge (com `ARC_API_KEY`, fora do plano):** re-rodar `vc33`/`ls20` e conferir que a fração de efeitos `structural` cai e ações HUD-only viram `none`.
