# Grounding de avatar na reward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Injetar o avatar aprendido (`MovementModel.avatar_id()`) no `_build_reward_prompt`, dizendo ao LLM que o objeto controlável é `state[K]`, para a reward medir a distância do avatar certo (não do `pts[0]` chutado).

**Architecture:** Só o método `_build_reward_prompt` muda: mapeia `avatar_id()` → índice no `state` e insere linha de grounding + ajusta o few-shot de distância pra usar `pts[K]`. Fallback pro comportamento atual quando não há avatar aprendido.

**Tech Stack:** Python 3.12, pytest, numpy.

## Global Constraints

- Default-safe: mesmo caminho já sob `CAUSAL_LLM`, sem toggle novo.
- Sem mudança em `_spatial_context`, `state`, `accept_reward`, `static_reward_check`, pilha de decisão.
- `self._move` (MovementModel) já existe (agent.py:86) e é populado no close-loop.
- `avatar_id()` retorna o `o.id` mais-movido ou `None` (cold-start). Ordem de `scene.objects` == ordem do `state`.
- Suíte verde ao fim (base 378 + novos).

---

### Task 1: Grounding do avatar em `_build_reward_prompt`

**Files:**
- Modify: `agents/causal/agent.py` — método `_build_reward_prompt` (linhas 481-507)
- Test: `tests/causal/test_agent_avatar_grounding.py` (criar)

**Interfaces:**
- Consumes: `self._move.avatar_id()`, `scene.objects`, `_spatial_context` (inalterado).
- Produces: `_build_reward_prompt(scene) -> str` com linha de grounding + few-shot usando `pts[K]`.

- [ ] **Step 1: Write the failing test**

Criar `tests/causal/test_agent_avatar_grounding.py`:

```python
import numpy as np

from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects


def _scene(coords):
    g = np.zeros((16, 16), dtype=int)
    for (r, c) in coords:
        g[r, c] = 3
    return match_objects(None, parse(g))


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


def test_prompt_uses_learned_avatar_index(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (0, 8)])
    objs = list(scene.objects)
    a._move.avatar_counts = {objs[1].id: 5}     # forca o avatar a ser o indice 1
    p = a._build_reward_prompt(scene)
    assert "state[1]" in p
    assert "pts[1]" in p
    assert "OBJETO CONTROLAVEL" in p


def test_prompt_fallback_when_no_avatar(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1")
    scene = _scene([(0, 0), (0, 8)])
    p = a._build_reward_prompt(scene)                # avatar_counts vazio -> None
    assert "OBJETO CONTROLAVEL" not in p
    assert "a=pts[0]" in p


def test_grounded_distance_reward_accepted(monkeypatch):
    import json
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    for coords in ([(0, 0), (0, 8)], [(0, 0), (0, 4)], [(0, 0), (0, 1)]):
        a._buffer.append((_scene(coords), "ACTION1", "structural"))
    body = ('pts=[b for _,b in state]\n'
            '    if len(pts)<2: return (0.0, False)\n'
            '    a=pts[1]\n'
            '    others=[c for k,c in enumerate(pts) if k!=1]\n'
            '    d=min(abs(a["x"]-c["x"])+abs(a["y"]-c["y"]) for c in others)\n'
            '    return (-float(d), d==0)')
    src = json.dumps({"type": "code", "source": "def reward_function(state):\n    " + body})
    a._llm = _Seq([src])
    ok = a._try_learn_reward(_scene([(0, 0), (0, 6)]))
    assert ok is True
    assert a._reward_fn is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/avatar-grounding && uv run pytest tests/causal/test_agent_avatar_grounding.py -v`
Expected: `test_prompt_uses_learned_avatar_index` FALHA (o prompt atual não tem `state[1]`/`pts[1]`/`OBJETO CONTROLAVEL`). Os outros 2 podem já passar.

- [ ] **Step 3: Write minimal implementation**

Substituir o método `_build_reward_prompt` (agent.py 481-507) por:

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
        ai = avatar_idx if avatar_idx is not None else 0
        ground = ""
        if avatar_idx is not None:
            ground = (f"OBJETO CONTROLAVEL (avatar) = state[{avatar_idx}]; a reward DEVE "
                      f"medir a distancia DELE (state[{avatar_idx}]) ate o alvo.\n")
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
            f"  # distancia: o avatar (state[{ai}]) se aproxima do alvo mais proximo\n"
            "  def reward_function(state):\n"
            "      pts=[b for _,b in state]\n"
            "      if len(pts)<2: return (0.0, False)\n"
            f"      a=pts[{ai}]\n"
            f"      others=[c for k,c in enumerate(pts) if k!={ai}]\n"
            "      d=min(abs(a['x']-c['x'])+abs(a['y']-c['y']) for c in others)\n"
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

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/avatar-grounding && uv run pytest tests/causal/test_agent_avatar_grounding.py -v`
Expected: PASS (3 testes).

- [ ] **Step 5: Run full suite (regressão)**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/avatar-grounding && uv run pytest tests/causal tests/kaggle -q`
Expected: tudo verde. `test_agent_reward_spatial.py` (que checa `a=pts[0]` no fallback) continua passando porque sem avatar aprendido o fallback mantém `pts[0]`.

- [ ] **Step 6: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_avatar_grounding.py
git commit -m "feat: grounding do avatar na reward (avatar_id -> state[K] no prompt + few-shot)"
```

---

### Task 2: Regenerar notebooks

**Files:**
- Modify: `kaggle/submission.ipynb`, `kaggle/offline.ipynb` (regenerados)

- [ ] **Step 1: Regenerar**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/avatar-grounding && uv run python kaggle/build_notebook.py && uv run python kaggle/build_offline_notebook.py`

- [ ] **Step 2: Verificar embed**

Run (verifica que o agent.py embutido tem o grounding):
```bash
python3 -c "
import json, base64, re
for nb in ['kaggle/submission.ipynb','kaggle/offline.ipynb']:
    d=json.load(open(nb)); src=''
    for c in d['cells']:
        s=''.join(c['source'])
        if 'FILES = ' in s: src=s
    files=json.loads(re.search(r'FILES = (\{.*?\})\n', src, re.S).group(1))
    a=base64.b64decode(files['agents/causal/agent.py']).decode()
    print(nb, 'avatar grounding:', 'OBJETO CONTROLAVEL' in a)
"
```
Expected: `True` nos dois.

- [ ] **Step 3: Commit**

```bash
git add kaggle/submission.ipynb kaggle/offline.ipynb
git commit -m "build: regen notebooks com grounding do avatar"
```

---

## Self-Review

**Spec coverage:** avatar_id→índice (Task 1 step 3) ✓; linha de grounding ✓; few-shot usa `pts[K]` ✓; fallback quando None ✓; testes 1-3 ✓; notebooks (Task 2) ✓.

**Placeholder scan:** sem TBD/TODO; código completo. ✓

**Type consistency:** `avatar_idx`/`ai` inteiros; `pts[{ai}]` no few-shot; `state[{avatar_idx}]` na linha de grounding. `avatar_id()` retorna `o.id` comparado com `o.id`. ✓

**Nota:** no teste, `a._move.avatar_counts = {objs[1].id: 5}` força `avatar_id()=objs[1].id` → índice 1 (mesmo objeto `scene`). O few-shot fallback mantém `a=pts[0]` (regressão do `test_agent_reward_spatial.py` intacta).
