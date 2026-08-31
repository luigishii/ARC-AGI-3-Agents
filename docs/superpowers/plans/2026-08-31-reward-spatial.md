# Reward espacial — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fazer o prompt de síntese de reward expor a estrutura espacial (posição x,y, distâncias par-a-par, resumo de grid) que o `state` já carrega, para o LLM sintetizar rewards espaciais em vez de contagem-de-cor.

**Architecture:** Adiciona uma função pura `_spatial_context(objects) -> str` em `agents/causal/agent.py` (bloco textual: objetos com x,y + distâncias Manhattan par-a-par + grid 3×3 coarse). Reescreve o método `_build_reward_prompt` para usar esse bloco + few-shot de rewards espaciais + corrigir a inconsistência do `size`. Nada mais muda: `state`, `compile_reward`, `accept_reward` e a pilha de decisão ficam intactos.

**Tech Stack:** Python 3.12, pytest, numpy. Sem libs novas.

## Global Constraints

- Mudança **default-safe**: roda no mesmo caminho que já roda sob `CAUSAL_LLM` — **sem novo toggle**.
- Contrato do `state` **inalterado**: `list[(shape_hash, {"x","y","color","shape"})]`.
- Não mexer em `compile_reward`, `accept_reward`, `static_reward_check`, nem na pilha de decisão.
- Suíte 100% verde ao fim (base 371 testes + os novos).
- Objeto de percepção: `o.centroid` é `(row, col)` → `x=int(round(o.centroid[1]))`, `y=int(round(o.centroid[0]))`; tem `o.color`, `o.size`.
- Few-shot de reward deve obedecer o `static_reward_check`: usar `state`, sem `import`, sem globals.

---

### Task 1: Helper `_spatial_context(objects) -> str`

**Files:**
- Modify: `agents/causal/agent.py` (adicionar função pura no nível de módulo, perto de `_obj_state` na linha ~36)
- Test: `tests/causal/test_agent_reward_spatial.py` (criar)

**Interfaces:**
- Consumes: objetos de percepção (`scene.objects`) — atributos `.centroid (row,col)`, `.color`, `.size`.
- Produces: `_spatial_context(objects) -> str` — bloco textual com 3 seções: `OBJETOS` (id,color,x,y,size), `DISTANCIAS` (Manhattan par-a-par, até 10 pares), `GRID 3x3` (cor por célula, `.`=vazio).

- [ ] **Step 1: Write the failing test**

Criar `tests/causal/test_agent_reward_spatial.py`:

```python
import numpy as np

from agents.causal.agent import _spatial_context
from agents.causal.perception import parse, match_objects


def _scene(coords):
    """coords: lista de (row, col) onde plantar um pixel cor 3 (objetos isolados)."""
    g = np.zeros((16, 16), dtype=int)
    for (r, c) in coords:
        g[r, c] = 3
    return match_objects(None, parse(g))


def test_spatial_context_has_xy_per_object():
    scene = _scene([(0, 0), (10, 12)])
    txt = _spatial_context(scene.objects)
    assert "x=0" in txt and "y=0" in txt          # objeto em (row0,col0)
    assert "x=12" in txt and "y=10" in txt         # objeto em (row10,col12)


def test_spatial_context_has_pairwise_distance():
    scene = _scene([(0, 0), (10, 12)])
    txt = _spatial_context(scene.objects)
    assert "DISTANCIAS" in txt
    assert "=22" in txt                            # |0-12|+|0-10| = 22


def test_spatial_context_has_grid_summary():
    scene = _scene([(0, 0), (10, 12)])
    txt = _spatial_context(scene.objects)
    assert "GRID 3x3" in txt


def test_spatial_context_single_object_no_distance_section():
    scene = _scene([(5, 5)])
    txt = _spatial_context(scene.objects)
    assert "x=5" in txt and "y=5" in txt
    assert "DISTANCIAS" not in txt                 # <2 objetos → sem seção de distância
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-spatial && uv run pytest tests/causal/test_agent_reward_spatial.py -v`
Expected: FAIL com `ImportError: cannot import name '_spatial_context'`.

- [ ] **Step 3: Write minimal implementation**

Em `agents/causal/agent.py`, logo após `_obj_state` (linha ~39), adicionar:

```python
def _spatial_context(objects) -> str:
    """Bloco textual com posição (x,y), distâncias par-a-par (Manhattan) e resumo de
    grid 3x3 dos objetos — pro prompt de síntese de reward enxergar estrutura espacial
    (não só cor). x=col, y=row (mesma convenção de _obj_state)."""
    objs = list(objects)[:8]
    pts = [(int(round(o.centroid[1])), int(round(o.centroid[0]))) for o in objs]
    lines = ["OBJETOS (x=col, y=row):"]
    for i, o in enumerate(objs):
        lines.append(f"  id={i} color={int(o.color)} x={pts[i][0]} y={pts[i][1]} size={o.size}")
    pairs = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
            pairs.append((d, i, j))
    if pairs:
        lines.append("DISTANCIAS (Manhattan):")
        for d, i, j in pairs[:10]:
            lines.append(f"  d(id{i},id{j})={d}")
    G = 3
    cell = {}
    for i, o in enumerate(objs):
        gx = min(pts[i][0] * G // 64, G - 1)
        gy = min(pts[i][1] * G // 64, G - 1)
        cell.setdefault((gy, gx), int(o.color))
    lines.append("GRID 3x3 (cor do objeto por celula, '.'=vazio):")
    for gy in range(G):
        lines.append("  " + " ".join(str(cell.get((gy, gx), ".")) for gx in range(G)))
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-spatial && uv run pytest tests/causal/test_agent_reward_spatial.py -v`
Expected: PASS (4 testes).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_reward_spatial.py
git commit -m "feat: _spatial_context (posicao+distancias+grid) para o prompt de reward"
```

---

### Task 2: Reescrever `_build_reward_prompt` com contexto espacial + few-shot

**Files:**
- Modify: `agents/causal/agent.py` — método `_build_reward_prompt` (linhas ~451-459)
- Test: `tests/causal/test_agent_reward_spatial.py` (acrescentar)

**Interfaces:**
- Consumes: `_spatial_context(objects)` da Task 1; `_try_learn_reward`, `accept_reward`, `static_reward_check` existentes.
- Produces: `_build_reward_prompt(scene) -> str` agora contendo o bloco espacial + few-shot de reward por distância e por arranjo. Sem mudança de assinatura.

- [ ] **Step 1: Write the failing test**

Acrescentar em `tests/causal/test_agent_reward_spatial.py`:

```python
from agents.causal.agent import CausalObjectAgent


class _Seq:
    def __init__(self, canned):
        self.canned = list(canned); self.calls = 0
    def complete(self, prompt):
        r = self.canned[min(self.calls, len(self.canned) - 1)]; self.calls += 1; return r


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_reward_prompt_shows_positions_and_fewshot(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (10, 12)])
    p = a._build_reward_prompt(scene)
    assert "x=0" in p and "y=10" in p              # posição exposta
    assert "DISTANCIAS" in p                        # distâncias expostas
    assert "manhattan" in p.lower() or "x']" in p   # few-shot espacial presente


def test_spatial_reward_is_accepted(monkeypatch):
    # reward de distância (gradiente real entre cenas com objetos em posições diferentes)
    import json
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    for coords in ([(0, 0), (0, 8)], [(0, 0), (0, 4)], [(0, 0), (0, 1)]):
        a._buffer.append((_scene(coords), "ACTION1", "structural"))
    body = ('pts=[b for _,b in state]\n'
            '    if len(pts)<2: return (0.0, False)\n'
            '    a=pts[0]\n'
            '    d=min(abs(a["x"]-b["x"])+abs(a["y"]-b["y"]) for b in pts[1:])\n'
            '    return (-float(d), d==0)')
    src = json.dumps({"type": "code", "source": "def reward_function(state):\n    " + body})
    a._llm = _Seq([src])
    ok = a._try_learn_reward(_scene([(0, 0), (0, 6)]))
    assert ok is True
    assert a._reward_fn is not None


def test_constant_reward_still_rejected(monkeypatch):
    import json
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    for coords in ([(0, 0)], [(0, 2)], [(0, 4)]):
        a._buffer.append((_scene(coords), "ACTION1", "structural"))
    src = json.dumps({"type": "code",
                      "source": "def reward_function(state):\n    _ = len(state)\n    return (0, False)"})
    a._llm = _Seq([src])
    ok = a._try_learn_reward(_scene([(0, 0)]))
    assert ok is False
    assert a._reward_fn is None
    assert a._reward_rejected >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-spatial && uv run pytest tests/causal/test_agent_reward_spatial.py -v`
Expected: `test_reward_prompt_shows_positions_and_fewshot` FALHA (prompt atual não tem x/DISTANCIAS/few-shot). Os outros dois podem já passar (o wiring de aceitação já existe) — tudo bem.

- [ ] **Step 3: Write minimal implementation**

Substituir o método `_build_reward_prompt` (agent.py ~451-459) por:

```python
    def _build_reward_prompt(self, scene) -> str:
        ctx = _spatial_context(scene.objects)
        return (
            "Infira reward_function(state) que retorna (reward, goal_flag). REGRAS: "
            "(1) reward é um número GRADUADO — maior = mais perto de resolver, NÃO use só 0/1; "
            "(2) NÃO hardcode tamanhos/posições exatos (magic numbers) — use relações/distâncias; "
            "(3) goal_flag=True SÓ quando o nível está realmente resolvido (raro). "
            "A META costuma ser ESPACIAL: aproximar/alinhar um objeto de um alvo, ou casar um "
            "arranjo de células — use POSIÇÃO (x,y) e DISTÂNCIAS, não contagem de cor. "
            "O state é lista de (tipo,{x,y,color,shape}); x=col, y=row.\n"
            f"{ctx}\n"
            "EXEMPLOS (reward espacial, só usam o state, sem import):\n"
            "  # distancia: objeto[0] se aproxima do alvo mais proximo\n"
            "  def reward_function(state):\n"
            "      pts=[b for _,b in state]\n"
            "      if len(pts)<2: return (0.0, False)\n"
            "      a=pts[0]\n"
            "      d=min(abs(a['x']-b['x'])+abs(a['y']-b['y']) for b in pts[1:])\n"
            "      return (-float(d), d==0)\n"
            "  # arranjo: quantos objetos alinhados na mesma linha (y) do alvo\n"
            "  def reward_function(state):\n"
            "      pts=[b for _,b in state]\n"
            "      if not pts: return (0.0, False)\n"
            "      ys=[b['y'] for b in pts]\n"
            "      return (float(sum(1 for y in ys if y==ys[0])), False)\n"
            'Responda SÓ JSON {"type":"code","source":"def reward_function(state): ..."}'
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-spatial && uv run pytest tests/causal/test_agent_reward_spatial.py -v`
Expected: PASS (7 testes no arquivo).

- [ ] **Step 5: Run full suite (regressão)**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-spatial && uv run pytest tests/causal tests/kaggle -q`
Expected: tudo verde (base + novos). Em especial `tests/causal/test_agent_reward_hardening.py` continua passando (não quebramos o hardening).

- [ ] **Step 6: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_reward_spatial.py
git commit -m "feat: _build_reward_prompt expoe posicao/distancia/grid + few-shot espacial"
```

---

## Self-Review

**Spec coverage:**
- Objetos com posição → Task 1 `OBJETOS` + Task 2 prompt. ✓
- Distâncias par-a-par → Task 1 `DISTANCIAS`. ✓
- Resumo de grid → Task 1 `GRID 3x3`. ✓
- Guia + few-shot espacial → Task 2. ✓
- Corrigir inconsistência do `size` → Task 2 (o novo prompt mostra `size` ao lado de x,y via `_spatial_context`, não isolado). ✓
- Manter REGRAS + `accept_reward` → Task 2 mantém as 3 regras; teste de regressão `test_constant_reward_still_rejected`. ✓
- Contrato do `state` inalterado → nenhuma task toca `_obj_state`/`compile_reward`. ✓

**Placeholder scan:** sem TBD/TODO; todo código está escrito. ✓

**Type consistency:** `_spatial_context(objects)` (Task 1) é chamado com `scene.objects` em `_build_reward_prompt` (Task 2). `pts[i]` é `(x,y)`. Convenção x=col/y=row consistente com `_obj_state`. ✓

**Nota de execução:** confirmar que `_scene`/`match_objects` produzem objetos isolados para as coordenadas dadas (pixels não-adjacentes → componentes separados). As coords dos testes são espaçadas o bastante (dist ≥ 4) para não mergear.
