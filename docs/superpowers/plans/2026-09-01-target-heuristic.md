# Alvo por heurística na reward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Injetar um ALVO PROVÁVEL (heurística cor-rara+compacto) no `_build_reward_prompt` + few-shot explícito avatar→alvo, para a reward parar de mirar o alvo errado.

**Architecture:** Novo helper puro `_pick_target(objects, avatar_idx)` em `agents/causal/agent.py`; `_build_reward_prompt` passa a computar `target_idx` e usar um few-shot explícito `a=pts[K]; t=pts[T]`. Fallback pro comportamento atual quando não há avatar.

**Tech Stack:** Python 3.12, pytest.

## Global Constraints

- Default-safe (mesmo caminho sob `CAUSAL_LLM`, sem toggle).
- Sem mudança em `_spatial_context`, `state`, `accept_reward`, pilha.
- `Object`: `.color`, `.size`, `.bbox=(r0,c0,r1,c1)`, `.centroid=(row,col)`, `.id`.
- Suíte verde (base 381 + novos).

---

### Task 1: `_pick_target` + few-shot explícito avatar→alvo

**Files:**
- Modify: `agents/causal/agent.py` (novo `_pick_target` no módulo; reescreve `_build_reward_prompt`)
- Test: `tests/causal/test_agent_target_heuristic.py` (criar)

**Interfaces:**
- Produces: `_pick_target(objects, avatar_idx) -> int|None`; `_build_reward_prompt` com hint de alvo + few-shot explícito.

- [ ] **Step 1: Write the failing test**

Criar `tests/causal/test_agent_target_heuristic.py`:

```python
from types import SimpleNamespace as NS

import numpy as np

from agents.causal.agent import _pick_target, CausalObjectAgent
from agents.causal.perception import parse, match_objects


def _obj(color, size, bbox, centroid):
    return NS(color=color, size=size, bbox=bbox, centroid=centroid)


def test_pick_target_rare_compact():
    objs = [
        _obj(9, 15, (0, 0, 2, 4), (1, 2)),        # 0 avatar
        _obj(2, 400, (10, 10, 30, 30), (20, 20)),  # 1 fundo (maior) -> excluido
        _obj(5, 10, (5, 0, 5, 9), (5, 4)),        # 2 barra 1x10 -> excluida
        _obj(4, 4, (8, 8, 9, 9), (8, 8)),         # 3 comum
        _obj(4, 4, (8, 20, 9, 21), (8, 20)),      # 4 comum
        _obj(7, 4, (2, 2, 3, 3), (2, 2)),         # 5 raro compacto
    ]
    assert _pick_target(objs, 0) == 5


def test_pick_target_excludes_bar():
    objs = [
        _obj(9, 15, (0, 0, 2, 4), (1, 2)),        # avatar
        _obj(5, 10, (5, 0, 5, 9), (5, 4)),        # barra raro-mas-alongada -> excluida
        _obj(4, 4, (8, 8, 9, 9), (8, 8)),
        _obj(4, 4, (8, 20, 9, 21), (8, 20)),
        _obj(4, 4, (8, 30, 9, 31), (8, 30)),
    ]
    assert _pick_target(objs, 0) in (2, 3, 4)


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


def _scene(coords):
    g = np.zeros((16, 16), dtype=int)
    for (r, c) in coords:
        g[r, c] = 3
    return match_objects(None, parse(g))


def test_prompt_has_target_hint_and_explicit_fewshot(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (0, 8)])
    objs = list(scene.objects)
    a._move.avatar_counts = {objs[0].id: 5}       # avatar = indice 0
    p = a._build_reward_prompt(scene)
    assert "ALVO PROVAVEL = state[" in p
    assert "t=pts[" in p                           # few-shot explicito avatar->alvo


def test_prompt_fallback_when_no_avatar(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (0, 8)])
    p = a._build_reward_prompt(scene)
    assert "ALVO PROVAVEL" not in p
    assert "a=pts[0]" in p
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/target-heuristic && uv run pytest tests/causal/test_agent_target_heuristic.py -q`
Expected: FAIL (`_pick_target` não existe / `ALVO PROVAVEL` ausente).

- [ ] **Step 3: Implement `_pick_target` (após `_spatial_context` no módulo)**

```python
def _pick_target(objects, avatar_idx):
    """Alvo provável: objeto não-avatar, cor rara, compacto (não fundo/barra). Índice ou None."""
    objs = list(objects)[:8]
    cand = [i for i in range(len(objs)) if i != avatar_idx]
    if not cand:
        return None
    color_count = {}
    for o in objs:
        color_count[o.color] = color_count.get(o.color, 0) + 1
    largest = max(cand, key=lambda i: objs[i].size)
    cand2 = [i for i in cand if i != largest] or cand

    def _elongated(i):
        r0, c0, r1, c1 = objs[i].bbox
        w, h = c1 - c0 + 1, r1 - r0 + 1
        return max(w, h) / max(1, min(w, h)) >= 4

    cand3 = [i for i in cand2 if not _elongated(i)] or cand2
    av = objs[avatar_idx]
    ax, ay = av.centroid[1], av.centroid[0]

    def _key(i):
        o = objs[i]
        dist = abs(o.centroid[1] - ax) + abs(o.centroid[0] - ay)
        return (color_count[o.color], o.size, dist)

    return min(cand3, key=_key)
```

- [ ] **Step 4: Reescrever `_build_reward_prompt`**

Substituir o método por (mantém REGRAS/ctx/arranjo; muda a parte avatar/alvo e o 1º few-shot):

```python
    def _build_reward_prompt(self, scene) -> str:
        ctx = _spatial_context(scene.objects)
        aid = self._move.avatar_id()
        avatar_idx = None
        if aid is not None:
            for i, o in enumerate(list(scene.objects)[:8]):
                if o.id == aid:
                    avatar_idx = i
                    break
        target_idx = _pick_target(scene.objects, avatar_idx) if avatar_idx is not None else None
        ai = avatar_idx if avatar_idx is not None else 0
        if avatar_idx is not None and target_idx is not None:
            ground = (f"OBJETO CONTROLAVEL (avatar) = state[{avatar_idx}]. "
                      f"ALVO PROVAVEL = state[{target_idx}] (objeto raro/compacto, distinto do avatar). "
                      f"A reward DEVE medir a distancia state[{avatar_idx}] -> state[{target_idx}].\n")
            few = (
                f"  # avatar (state[{avatar_idx}]) se aproxima do alvo (state[{target_idx}])\n"
                "  def reward_function(state):\n"
                "      pts=[b for _,b in state]\n"
                f"      if len(pts)<={max(avatar_idx, target_idx)}: return (0.0, False)\n"
                f"      a=pts[{avatar_idx}]; t=pts[{target_idx}]\n"
                "      d=abs(a['x']-t['x'])+abs(a['y']-t['y'])\n"
                "      return (-float(d), d==0)\n"
            )
        elif avatar_idx is not None:
            ground = (f"OBJETO CONTROLAVEL (avatar) = state[{avatar_idx}]; a reward DEVE "
                      f"medir a distancia DELE (state[{avatar_idx}]) ate o alvo.\n")
            few = (
                f"  # distancia: o avatar (state[{ai}]) se aproxima do alvo mais proximo\n"
                "  def reward_function(state):\n"
                "      pts=[b for _,b in state]\n"
                "      if len(pts)<2: return (0.0, False)\n"
                f"      a=pts[{ai}]\n"
                f"      others=[c for k,c in enumerate(pts) if k!={ai}]\n"
                "      d=min(abs(a['x']-c['x'])+abs(a['y']-c['y']) for c in others)\n"
                "      return (-float(d), d==0)\n"
            )
        else:
            ground = ""
            few = (
                "  # distancia: objeto[0] se aproxima do alvo mais proximo\n"
                "  def reward_function(state):\n"
                "      pts=[b for _,b in state]\n"
                "      if len(pts)<2: return (0.0, False)\n"
                "      a=pts[0]\n"
                "      d=min(abs(a['x']-b['x'])+abs(a['y']-b['y']) for b in pts[1:])\n"
                "      return (-float(d), d==0)\n"
            )
        return (
            "Infira reward_function(state) que retorna (reward, goal_flag). REGRAS: "
            "(1) reward é um número GRADUADO — maior = mais perto de resolver, NÃO use só 0/1; "
            "(2) NÃO hardcode tamanhos/posições exatos (magic numbers) — use relações/distâncias; "
            "(3) goal_flag=True SÓ quando o nível está realmente resolvido (raro). "
            "A META costuma ser ESPACIAL: aproximar/alinhar um objeto de um alvo, ou casar um "
            "arranjo de células — use POSIÇÃO (x,y) e DISTÂNCIAS, não contagem de cor. "
            "O state é lista de (tipo,{x,y,color,shape}); x=col, y=row.\n"
            f"{ctx}\n"
            f"{ground}"
            "EXEMPLOS (reward espacial, só usam o state, sem import):\n"
            f"{few}"
            "  # arranjo: quantos objetos alinhados na mesma linha (y) do alvo\n"
            "  def reward_function(state):\n"
            "      pts=[b for _,b in state]\n"
            "      if not pts: return (0.0, False)\n"
            "      ys=[b['y'] for b in pts]\n"
            "      return (float(sum(1 for y in ys if y==ys[0])), False)\n"
            'Responda SÓ JSON {"type":"code","source":"def reward_function(state): ..."}'
        )
```

- [ ] **Step 5: Run new tests (green)**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/target-heuristic && uv run pytest tests/causal/test_agent_target_heuristic.py -q`
Expected: PASS (4 testes).

- [ ] **Step 6: Full suite (regressão)**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/target-heuristic && uv run pytest tests/causal tests/kaggle -q`
Expected: verde. `test_agent_avatar_grounding.py` (state[1]/pts[1]) e `test_agent_reward_spatial.py` (pts[0] fallback) continuam passando: avatar-primado com 2 objetos → avatar=1, target=0 → prompt tem `state[1]` e `pts[1]`; sem avatar → else → `a=pts[0]`.

- [ ] **Step 7: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_target_heuristic.py
git commit -m "feat: alvo por heuristica (cor rara+compacto) + few-shot explicito avatar->alvo na reward"
```

---

### Task 2: Regenerar notebooks

- [ ] **Step 1:** `cd .../target-heuristic && uv run python kaggle/build_notebook.py && uv run python kaggle/build_offline_notebook.py`
- [ ] **Step 2:** Verificar embed (`ALVO PROVAVEL` no agent.py embutido dos 2 .ipynb).
- [ ] **Step 3:** `git add kaggle/*.ipynb && git commit -m "build: regen notebooks com alvo por heuristica"`

---

## Self-Review

**Spec coverage:** heurística cor-rara+compacto (`_pick_target`, exclui maior+alongado, rarest→size→dist) ✓; hint no prompt ✓; few-shot explícito avatar→alvo ✓; fallback sem avatar ✓; testes 1-5 ✓; notebooks ✓.

**Placeholder scan:** sem TBD; código completo. ✓

**Type consistency:** `_pick_target(objects, avatar_idx)->int|None`; usado com `scene.objects` + `avatar_idx`. `bbox=(r0,c0,r1,c1)`, `centroid=(row,col)`. few-shot indexa `pts[avatar_idx]`/`pts[target_idx]`. ✓

**Nota:** teste `_pick_target` usa stubs `SimpleNamespace` (só `.color/.size/.bbox/.centroid`) — determinístico, sem depender de match_objects.
