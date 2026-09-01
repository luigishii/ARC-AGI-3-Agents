# Direct Reasoning Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar uma política de raciocínio direto passo-a-passo em que o LLM escolhe a próxima ação imediata a cada frame (gated por cooldown), a partir da cena objeto-cêntrica + `available_actions` + feedback da última ação, validada/executada pela fundação causal.

**Architecture:** Novo builder `build_direct_prompt` em `llm.py` (orientado a AÇÃO, reusa `parse_goal`/`execute_goal`); nova camada `_direct_decide` no topo da pilha de decisão de `agent.py` sob toggle `CAUSAL_DIRECT`; quando ligada, substitui a inferência-de-meta persistente como driver do LLM. Default-off → zero regressão.

**Tech Stack:** Python 3.12, pytest, numpy, `arcengine` (stubs de teste). Sem novas dependências.

## Global Constraints

- Toggle novo `CAUSAL_DIRECT` default **off** (`os.environ.get("CAUSAL_DIRECT","0") != "0"`) → caminho existente idêntico quando unset.
- `CAUSAL_DIRECT_COOLDOWN` default **2** (`int(os.environ.get("CAUSAL_DIRECT_COOLDOWN","2"))`).
- Reusar `parse_goal` e `execute_goal` de `agents/causal/llm.py` — NÃO reimplementar parsing/execução.
- `_direct_decide` é **exception-safe**: qualquer falha → `None` → fall-through determinístico. Nenhuma exceção propaga.
- Uma alavanca por vez: nenhuma mudança nas camadas navigate/IW/rprog/η/cover além do gate condicional do bloco de meta.
- Suíte inteira verde ao fim (393 + novos).
- Remote é upstream `arcprize` (read-only): **não** dar push/merge remoto. Integração é merge LOCAL na `main`.
- ARC_API_KEY nunca em arquivo versionado (N/A aqui — nada toca chave/rede).

---

### Task 1: `build_direct_prompt` em `llm.py`

**Files:**
- Modify: `agents/causal/llm.py` (adicionar função + constante `_DIRECT_INSTRUCTION` após `_INSTRUCTION`)
- Test: `tests/causal/test_llm_direct.py` (criar)

**Interfaces:**
- Consumes: `scene.objects` (cada obj tem `.id`, `.color`, `.centroid`, `.size`, `.bbox`); `dyn: dict` com chave `"available"` (lista); `last: dict | None` com chaves `"key"` e `"effect"`.
- Produces: `build_direct_prompt(scene, dyn: dict, last: dict | None = None) -> str`.

- [ ] **Step 1: Write the failing tests**

Criar `tests/causal/test_llm_direct.py`:

```python
from types import SimpleNamespace

from agents.causal.llm import build_direct_prompt


def _obj(id=0, color=3):
    return SimpleNamespace(id=id, color=color, centroid=(1, 1), size=4, bbox=(1, 1, 2, 2))


def _scene(*objs):
    return SimpleNamespace(objects=list(objs))


def test_direct_prompt_lists_available():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1", "ACTION2"]})
    assert "AVAILABLE_ACTIONS" in p
    assert "ACTION1" in p and "ACTION2" in p


def test_direct_prompt_shows_objects():
    p = build_direct_prompt(_scene(_obj(id=0, color=3)), {"available": ["ACTION1"]})
    assert "OBJETOS" in p
    assert "color=3" in p


def test_direct_prompt_last_feedback_present():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1"]},
                            {"key": "ACTION2", "effect": "structural"})
    assert "ACTION2" in p and "structural" in p
    assert "PROGRESSO" in p


def test_direct_prompt_last_omitted_when_none():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1"]}, None)
    assert "ultima acao" not in p.lower()
    p2 = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1"]},
                             {"key": None, "effect": None})
    assert "ultima acao" not in p2.lower()


def test_direct_prompt_asks_single_action():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1"]})
    assert '"type":"press"' in p
    assert '"type":"click_cell"' in p
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/causal/test_llm_direct.py -v`
Expected: FAIL com `ImportError: cannot import name 'build_direct_prompt'`.

- [ ] **Step 3: Implement `build_direct_prompt`**

Em `agents/causal/llm.py`, após a constante `_INSTRUCTION` (que termina por volta da linha 20), adicionar:

```python
_DIRECT_INSTRUCTION = (
    "Escolha a PROXIMA acao imediata (uma so) para fazer progresso. Responda "
    "APENAS um JSON, sem markdown, sem prosa:\n"
    '{"type":"press","action":"ACTIONk"}   (k da lista disponivel)\n'
    '{"type":"click_cell","gx":0,"gy":0}   (gx,gy em 0..5 = celula do grid 6x6)'
)


def build_direct_prompt(scene, dyn, last=None) -> str:
    """Prompt orientado a ACAO (distinto de build_prompt, orientado a META): serializa
    a cena objeto-centrica + AVAILABLE_ACTIONS + feedback da ultima acao, e pede UMA
    proxima acao. O parsing reusa parse_goal; a execucao reusa execute_goal."""
    dyn = dyn or {}
    lines = [f"OBJETOS ({len(scene.objects)}):"]
    for o in scene.objects:
        lines.append(
            f"  id={o.id} color={o.color} centroid={o.centroid} "
            f"size={o.size} bbox={o.bbox}"
        )
    lines.append(f"AVAILABLE_ACTIONS: {dyn.get('available', [])}   (use SO essas)")
    if last and last.get("key"):
        eff = last.get("effect") or "nenhuma mudanca"
        lines.append(
            f"Sua ultima acao {last['key']} produziu: {eff}. Escolha a PROXIMA "
            "acao que faz PROGRESSO; NAO repita uma acao que nao mudou nada."
        )
    lines.append(_DIRECT_INSTRUCTION)
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/causal/test_llm_direct.py -v`
Expected: PASS (5 testes).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/llm.py tests/causal/test_llm_direct.py
git commit -m "feat: build_direct_prompt (score-max Lever #2 — prompt orientado a acao)"
```

---

### Task 2: `_direct_decide` + estado + wiring + stats em `agent.py`

**Files:**
- Modify: `agents/causal/agent.py` (import linha 20; estado em `_init_causal_state`; wiring em `choose_action`; novo método `_direct_decide`; `phase2_stats`)
- Test: `tests/causal/test_agent_direct.py` (criar)

**Interfaces:**
- Consumes: `build_direct_prompt` (Task 1); `parse_goal`, `execute_goal` (já importados); `self._llm.complete`, `self._llm_calls`, `self._llm_max`, `self._last_key`, `keymap` (dict key→Candidate), `avail`, `moves`.
- Produces: método `_direct_decide(self, scene, avail, keymap, moves) -> Candidate | None`; atributos `self._direct_on`, `self._direct_cooldown`, `self._since_direct`, `self._last_effect_kind`, `self._direct_calls`, `self._direct_hits`; chaves `"direct_calls"`/`"direct_hits"` em `phase2_stats()`.

- [ ] **Step 1: Write the failing tests**

Criar `tests/causal/test_agent_direct.py`:

```python
from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent


class _Fake:
    def __init__(self, canned):
        self.canned = canned
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.canned


class _Frame:
    def __init__(self, frame, available=None, levels=0):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = levels
        self.available_actions = available or [GameAction.ACTION1]
        self.full_reset = False


def _grid(v=3):
    g = [[0] * 8 for _ in range(8)]
    g[1][1] = v
    return [g]


def _agent(monkeypatch, **env):
    env.setdefault("CAUSAL_LLM", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def test_direct_uses_valid_action(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    act = a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert act.name == "ACTION1"
    assert a.phase2_stats()["direct_hits"] == 1
    assert a.phase2_stats()["direct_calls"] == 1


def test_direct_invalid_falls_through(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1")
    a._llm = _Fake('{"type":"press","action":"ACTION5"}')   # fora do available
    act = a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert act is not None                                   # nao crasha
    assert a.phase2_stats()["direct_hits"] == 0
    assert a.phase2_stats()["direct_calls"] == 1


def test_direct_off_never_queries(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="0")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert a.phase2_stats()["direct_calls"] == 0


def test_direct_cooldown(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1", CAUSAL_DIRECT_COOLDOWN="2")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    a.choose_action([], _Frame(_grid(3), available=[GameAction.ACTION1]))   # consulta
    a.choose_action([], _Frame(_grid(4), available=[GameAction.ACTION1]))   # cooldown -> nao
    assert a.phase2_stats()["direct_calls"] == 1


def test_direct_budget_exhausted(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1", CAUSAL_LLM_MAX_CALLS="0")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert a.phase2_stats()["direct_calls"] == 0


def test_direct_skips_persistent_goal(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1")
    a._llm = _Fake('{"type":"press","action":"ACTION1"}')
    a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION1]))
    assert a._goal is None     # bloco de meta-persistente pulado sob direct


def test_direct_click_cell(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_DIRECT="1")
    a._llm = _Fake('{"type":"click_cell","gx":2,"gy":3}')
    act = a.choose_action([], _Frame(_grid(), available=[GameAction.ACTION6]))
    assert act is not None
    assert a.phase2_stats()["direct_hits"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/causal/test_agent_direct.py -v`
Expected: FAIL — `KeyError: 'direct_hits'` (phase2_stats sem a chave) / `AttributeError` (`_direct_on`/`_direct_decide` inexistentes).

- [ ] **Step 3a: Import `build_direct_prompt`**

Em `agents/causal/agent.py` linha 20, trocar:

```python
from .llm import shared_llm_client, build_prompt, parse_goal, execute_goal, client_kind
```
por:
```python
from .llm import (shared_llm_client, build_prompt, build_direct_prompt,
                  parse_goal, execute_goal, client_kind)
```

- [ ] **Step 3b: Estado novo em `_init_causal_state`**

Em `agents/causal/agent.py`, logo após `self._repair_max = int(os.environ.get("CAUSAL_REPAIR", "1"))` (por volta da linha 124), adicionar:

```python
        # Score-max Lever #2: politica de raciocinio direto passo-a-passo
        self._direct_on = os.environ.get("CAUSAL_DIRECT", "0") != "0"
        self._direct_cooldown = int(os.environ.get("CAUSAL_DIRECT_COOLDOWN", "2"))
        self._since_direct = 10 ** 9      # como _since_query: consulta no 1o frame elegivel
        self._last_effect_kind = None     # efeito real da ultima acao (feedback do prompt)
        self._direct_calls = 0
        self._direct_hits = 0
```

- [ ] **Step 3c: Novo método `_direct_decide`**

Em `agents/causal/agent.py`, adicionar um método logo após `_iw_decide` (que termina por volta da linha 404):

```python
    def _direct_decide(self, scene, avail, keymap, moves):
        """Score-max Lever #2: consulta o LLM pela PROXIMA acao imediata (esparso via
        cooldown), valida contra keymap (available-guard) e devolve o candidato. Miss ou
        erro -> None (cai na pilha deterministica). Exception-safe."""
        if self._since_direct < self._direct_cooldown:
            return None
        if self._llm_calls >= self._llm_max:
            return None
        self._since_direct = 0
        self._llm_calls += 1
        self._direct_calls += 1
        dyn = {"available": [str(a) for a in avail], "moves": moves, "notes": ""}
        last = {"key": self._last_key, "effect": self._last_effect_kind}
        try:
            resp = self._llm.complete(build_direct_prompt(scene, dyn, last))
            g = parse_goal(resp)
            key = execute_goal(g, scene, moves) if g is not None else None
        except Exception:
            return None
        cand = keymap.get(key) if key is not None else None
        if cand is not None:
            self._direct_hits += 1
        return cand
```

- [ ] **Step 3d: Wiring em `choose_action` — feedback da última ação**

Em `agents/causal/agent.py`, no bloco de limpeza do topo (dentro do `if need_reset or getattr(latest_frame, "full_reset", False):`, junto às linhas `self._prev_scene = None` etc., ~linha 200-203), adicionar:

```python
            self._last_effect_kind = None
```

No fecha-loop, no `if level_up:` (por volta da linha 223-228), adicionar após `self._goal = None`:

```python
                self._last_effect_kind = None
```

E no `else:` (transição decisão→decisão, por volta da linha 229-239), adicionar (pode ser logo após a linha do `self._buffer.append(...)`):

```python
                self._last_effect_kind = actual.kind
```

- [ ] **Step 3e: Wiring — incremento do cooldown e gate do bloco de meta**

Em `agents/causal/agent.py`, após `self._since_query += 1` (linha 254), adicionar:

```python
        self._since_direct += 1
```

E na condição do bloco de consulta-meta persistente (linha 255-256), adicionar `and not self._direct_on`:

```python
        if (self._llm_on and self._goal is None and self._since_query >= QUERY_COOLDOWN
                and self._llm_calls < self._llm_max and not self._direct_on):
```

- [ ] **Step 3f: Wiring — camada direct no topo da pilha**

Em `agents/causal/agent.py`, a linha `cand = None` (linha 286) e o início do goal-path (linha 288 `if self._goal is not None:`) viram:

```python
        cand = None
        # (0) Score-max Lever #2: raciocinio direto passo-a-passo (topo da pilha)
        if self._direct_on:
            cand = self._direct_decide(scene, avail, keymap, moves)
        # (2) meta do LLM com validação
        if cand is None and self._goal is not None:
```

(As demais camadas `if cand is None and ...` seguem inalteradas.)

- [ ] **Step 3g: `phase2_stats` ganha as chaves**

Em `agents/causal/agent.py`, no dict retornado por `phase2_stats()` (por volta da linha 645, antes de `"eta_rows"`), adicionar:

```python
            "direct_calls": self._direct_calls,
            "direct_hits": self._direct_hits,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/causal/test_agent_direct.py -v`
Expected: PASS (7 testes).

- [ ] **Step 5: Run the full suite (no regression)**

Run: `uv run pytest tests/causal -q`
Expected: PASS (393 anteriores + 5 da Task 1 + 7 desta = 405), 0 falhas.

- [ ] **Step 6: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_direct.py
git commit -m "feat: camada de raciocinio direto no agent (CAUSAL_DIRECT + _direct_decide)"
```

---

### Task 3: Toggle nos notebooks + regenerar

**Files:**
- Modify: `kaggle/build_notebook.py` (constante `ENV`)
- Modify: `kaggle/build_offline_notebook.py` (constante `OFFLINE_ENV`)
- Regenerate: `kaggle/submission.ipynb`, `kaggle/offline.ipynb`
- Test: `tests/kaggle/test_build_notebook.py`, `tests/kaggle/test_build_offline_notebook.py` (adicionar 1 asserção cada)

**Interfaces:**
- Consumes: constantes `ENV` (build_notebook.py) e `OFFLINE_ENV` (build_offline_notebook.py).
- Produces: linha `CAUSAL_DIRECT=1` nos dois `.env` gerados.

- [ ] **Step 1: Write the failing tests**

Em `tests/kaggle/test_build_offline_notebook.py`, adicionar:

```python
def test_offline_env_has_direct():
    import kaggle.build_offline_notebook as b
    assert "CAUSAL_DIRECT=1" in b.OFFLINE_ENV
```

Em `tests/kaggle/test_build_notebook.py`, adicionar (usa `import kaggle.build_notebook as b`; se o arquivo já usa outro padrão de import do módulo, replicar o dele):

```python
def test_env_has_direct():
    import kaggle.build_notebook as b
    assert "CAUSAL_DIRECT=1" in b.ENV
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/kaggle/test_build_offline_notebook.py::test_offline_env_has_direct tests/kaggle/test_build_notebook.py::test_env_has_direct -v`
Expected: FAIL (a linha ainda não existe nos ENVs).

- [ ] **Step 3: Adicionar `CAUSAL_DIRECT=1` nos dois ENVs**

Em `kaggle/build_notebook.py`, na constante `ENV`, após a linha `"CAUSAL_FIX=1\n"` (linha 48), adicionar:

```python
    "CAUSAL_DIRECT=1\n"     # score-max Lever #2: raciocinio direto passo-a-passo
```

Em `kaggle/build_offline_notebook.py`, na constante `OFFLINE_ENV`, após a linha `"CAUSAL_FIX=1\n"`, adicionar a mesma linha:

```python
    "CAUSAL_DIRECT=1\n"     # score-max Lever #2: raciocinio direto passo-a-passo
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/kaggle -q`
Expected: PASS (todos os testes kaggle, incluindo os 2 novos).

- [ ] **Step 5: Regenerar os notebooks**

Run:
```bash
uv run python kaggle/build_notebook.py
uv run python kaggle/build_offline_notebook.py
```
Expected: imprime `wrote .../submission.ipynb ...` e `wrote .../offline.ipynb ...`.

- [ ] **Step 6: Verificar o embed base64 (o código novo entrou)**

Run:
```bash
uv run python -c "import json,base64,re; nb=json.load(open('kaggle/offline.ipynb')); src=''.join(''.join(c['source']) for c in nb['cells']); m=re.search(r'\"agents/causal/llm.py\":\s*\"([^\"]+)\"', src); print('build_direct_prompt' in base64.b64decode(m.group(1)).decode())"
```
Expected: `True` (a nova função está embarcada no `llm.py` base64 do notebook offline).

- [ ] **Step 7: Commit**

```bash
git add kaggle/build_notebook.py kaggle/build_offline_notebook.py kaggle/submission.ipynb kaggle/offline.ipynb tests/kaggle/test_build_notebook.py tests/kaggle/test_build_offline_notebook.py
git commit -m "build: CAUSAL_DIRECT=1 nos 2 builders + notebooks regenerados"
```

---

## Self-Review

**Spec coverage:**
- Componente 1 (`build_direct_prompt`) → Task 1. ✓
- Componente 2 (`_direct_decide`) → Task 2 Step 3c. ✓
- Componente 3 wiring (feedback última ação, cooldown, gate meta, camada topo) → Task 2 Steps 3d-3f. ✓
- Componente 4 (diag `direct_calls`/`direct_hits`, toggles notebooks) → Task 2 Step 3g + Task 3. ✓
- Constantes/estado (`_direct_on`/`_direct_cooldown`/`_since_direct`/`_last_effect_kind`/counters) → Task 2 Step 3b. ✓
- 7 testes FakeLLM da spec → Task 2 Step 1. ✓
- Anti-regressão (default-off) → Global Constraints + Task 2 Step 5 (suíte verde). ✓

**Placeholder scan:** nenhum TBD/TODO; todo passo tem código concreto.

**Type consistency:** `build_direct_prompt(scene, dyn, last=None)` idêntico em Task 1 (def) e Task 2 (uso via `_direct_decide`). `_direct_decide(self, scene, avail, keymap, moves)` — assinatura e chamada (Step 3f) batem. Chaves de stats `direct_calls`/`direct_hits` idênticas em Step 3b (init), 3c (incremento), 3g (stats) e nos testes. Toggle `CAUSAL_DIRECT`/`CAUSAL_DIRECT_COOLDOWN` consistente init↔testes↔notebooks.

**Nota de ambiguidade resolvida:** a assinatura do método é `_direct_decide(self, scene, avail, keymap, moves)` (sem o `cands` que a spec listava, pois só `keymap` é usado — evita param morto; consistente com o uso interno).
