# Endurecer a Síntese da Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rejeitar rewards sintetizadas patológicas (constante / sempre-True / que quebra) avaliando-as em estados reais observados, com prompt estrito e self-repair, para o LLM produzir uma reward graduada e discriminante.

**Architecture:** `accept_reward` em `goals.py` (aceitação comportamental sobre estados reais); `_try_learn_reward` reescrito para amostrar estados do `self._buffer`, validar com `static_reward_check` + `accept_reward`, e auto-reparar com o motivo da rejeição; `_build_reward_prompt` estrito; diagnóstico `reward_rejected`.

**Tech Stack:** Python 3.12, pytest. Só `agents/causal/goals.py` + `agents/causal/agent.py` + testes.

## Global Constraints

- Default-safe: aperto no caminho de síntese que já roda sob `CAUSAL_LLM`/`CAUSAL_TYPED`; sem novo toggle.
- Cold-start: `< min_states` estados → `accept_reward` aceita (bootstrap; preserva testes que não populam o buffer).
- À prova de exceção: `accept_reward` engole exceções da reward LLM-autorada (→ rejeita).
- Reusar o padrão de self-repair do f_τ (`_try_learn_type_rule`/`_build_repair_prompt`, `agent.py:456-486`).
- Manter a suíte verde (baseline na `main` `d97825c`; confirme com `pytest tests/causal tests/kaggle -q`).
- Não hardcodar `ARC_API_KEY` nem segredos.

---

### Task 1: `accept_reward` em goals.py

**Files:**
- Modify: `agents/causal/goals.py` (append após `value_fn_from_reward`, fim do arquivo)
- Test: `tests/causal/test_goals_accept_reward.py` (create)

**Interfaces:**
- Consumes: `compile_reward` (já em `goals.py`).
- Produces: `accept_reward(source, sample_states, min_states=3) -> (bool, str)`. `sample_states` = lista de estados, cada estado = lista de `(tipo, obj_dict)`. Aceita/rejeita por: não-compila, exceção em estado real, `all(goal_flag)` (falso-positivo), escalar constante entre estados distintos (sem gradiente); `< min_states` → aceita (cold-start).

- [ ] **Step 1: Write the failing tests**

Create `tests/causal/test_goals_accept_reward.py`:

```python
from agents.causal.goals import accept_reward

_ST3 = [[("h", {"color": 1, "size": 1})],
        [("h", {"color": 1, "size": 1}), ("h", {"color": 2, "size": 2})],
        [("h", {"color": 1, "size": 1}), ("h", {"color": 2, "size": 2}),
         ("h", {"color": 3, "size": 3})]]   # 3 estados distintos (len 1,2,3)


def test_reject_always_true():
    src = "def reward_function(state):\n    return (1, True)"
    ok, reason = accept_reward(src, _ST3)
    assert ok is False and "falso-positivo" in reason


def test_reject_constant_scalar():
    src = "def reward_function(state):\n    return (0, False)"   # constante em estados distintos
    ok, reason = accept_reward(src, _ST3)
    assert ok is False and "CONSTANTE" in reason


def test_accept_graded():
    src = "def reward_function(state):\n    return (len(state), False)"   # varia 1,2,3
    ok, reason = accept_reward(src, _ST3)
    assert ok is True


def test_reject_raises_on_real_state():
    src = "def reward_function(state):\n    return (1 / (len(state) - 1), False)"  # ZeroDiv em len==1
    ok, reason = accept_reward(src, _ST3)
    assert ok is False and "exceção" in reason


def test_reject_non_compiling():
    ok, reason = accept_reward("not python at all", _ST3)
    assert ok is False and "compila" in reason


def test_cold_start_accepts_few_states():
    src = "def reward_function(state):\n    return (0, False)"
    ok, reason = accept_reward(src, _ST3[:1])       # 1 estado < min_states
    assert ok is True


def test_identical_states_skip_gradient():
    same = [[("h", {"color": 1, "size": 1})]] * 3   # 3 estados IDÊNTICOS
    src = "def reward_function(state):\n    return (5, False)"   # constante, mas estados iguais
    ok, reason = accept_reward(src, same)
    assert ok is True                                # pula o teste de gradiente
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-hardening && uv run pytest tests/causal/test_goals_accept_reward.py -v`
Expected: FAIL — `ImportError: cannot import name 'accept_reward'`.

- [ ] **Step 3: Implement `accept_reward`**

In `agents/causal/goals.py`, append after `value_fn_from_reward` (end of file):

```python


def accept_reward(source, sample_states, min_states=3):
    """Aceitação COMPORTAMENTAL da reward: avalia em estados reais e rejeita patológicas.
    Retorna (aceito, motivo). Cold-start: < min_states estados -> aceita (bootstrap)."""
    fn = compile_reward(source)
    if fn is None:
        return (False, "não compila")
    if len(sample_states) < min_states:
        return (True, "poucos estados p/ julgar (cold-start)")
    vals, flags = [], []
    for st in sample_states:
        try:
            r = fn(st)
        except Exception:
            return (False, "levanta exceção em estado real")
        if isinstance(r, (tuple, list)) and len(r) >= 2:
            vals.append(float(r[0])); flags.append(bool(r[1]))
        else:
            vals.append(float(r)); flags.append(bool(r))
    if all(flags):
        return (False, "goal_flag=True em TODO estado (falso-positivo)")
    distinct_states = len({repr(st) for st in sample_states}) > 1
    if distinct_states and len({round(v, 6) for v in vals}) <= 1:
        return (False, "reward escalar CONSTANTE entre estados distintos (sem gradiente)")
    return (True, "ok")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-hardening && uv run pytest tests/causal/test_goals_accept_reward.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/goals.py tests/causal/test_goals_accept_reward.py
git commit -m "feat: accept_reward (aceitação comportamental da reward em estados reais)"
```

---

### Task 2: wiring em agent.py (síntese estrita + prompt + repair + diagnóstico)

**Files:**
- Modify: `agents/causal/agent.py` — import (linha 25); `_init_causal_state` (após `self._reward_src = None`, linha 96); `_build_reward_prompt` (linhas 407-412); `_try_learn_reward` (linhas 414-430); novo `_build_reward_repair_prompt`; `phase2_stats`
- Test: `tests/causal/test_agent_reward_hardening.py` (create)

**Interfaces:**
- Consumes: `accept_reward` (Task 1), `static_reward_check`/`compile_reward` (já importados), `parse_goal`, `_obj_state`, `self._buffer` (deque de `(scene, key, effect)`), `self._repair_max`, `self._n_samples`, `self._llm`.
- Produces: `self._reward_rejected: int`; `_try_learn_reward` aceita só reward que passa estático + comportamental, self-repara com o motivo; `_build_reward_repair_prompt(scene, err) -> str`; `phase2_stats` ganha `reward_rejected`.

- [ ] **Step 1: Write the failing tests**

Create `tests/causal/test_agent_reward_hardening.py`:

```python
import numpy as np

from agents.causal.agent import CausalObjectAgent
from agents.causal.perception import parse, match_objects


class _Seq:
    """FakeLLM que devolve respostas canned em sequência (1 por chamada)."""
    def __init__(self, canned):
        self.canned = list(canned)
        self.calls = 0

    def complete(self, prompt):
        r = self.canned[min(self.calls, len(self.canned) - 1)]
        self.calls += 1
        return r


def _agent(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.MAX_ACTIONS = 80
    a._init_causal_state()
    return a


def _scene(n):
    g = np.zeros((8, 8), dtype=int)
    for i in range(n):
        g[0, i * 2] = 3
    return match_objects(None, parse(g))


def _fill_buffer(a):
    for n in (1, 2, 3):                       # 3 cenas distintas → gradiente julgável
        a._buffer.append((_scene(n), "ACTION1", "structural"))


def _reward_json(body):
    import json
    return json.dumps({"type": "code",
                       "source": "def reward_function(state):\n    " + body})


# --- reward constante é rejeitada; _reward_rejected sobe; _reward_fn fica None ---
def test_rejects_constant_reward(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    _fill_buffer(a)
    a._llm = _Seq([_reward_json("return (0, False)")])
    ok = a._try_learn_reward(_scene(2))
    assert ok is False
    assert a._reward_fn is None
    assert a._reward_rejected >= 1


# --- reward graduada é aceita ---
def test_accepts_graded_reward(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="0")
    _fill_buffer(a)
    a._llm = _Seq([_reward_json("return (len(state), False)")])
    ok = a._try_learn_reward(_scene(2))
    assert ok is True
    assert a._reward_fn is not None


# --- self-repair: 1ª binária (rejeitada) -> 2ª graduada (aceita) ---
def test_self_repair_recovers(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_LLM="1", CAUSAL_REPAIR="1")
    _fill_buffer(a)
    a._llm = _Seq([_reward_json("return (0, False)"),        # rodada 1: rejeitada
                   _reward_json("return (len(state), False)")])  # rodada 2: aceita
    ok = a._try_learn_reward(_scene(2))
    assert ok is True
    assert a._reward_rejected >= 1


# --- prompt estrito tem as instruções-chave ---
def test_prompt_is_strict(monkeypatch):
    a = _agent(monkeypatch)
    p = a._build_reward_prompt(_scene(2))
    assert "GRADUADO" in p
    assert "magic" in p.lower() or "hardcode" in p.lower()
    assert "goal_flag=True" in p


# --- phase2_stats expõe reward_rejected ---
def test_phase2_has_reward_rejected(monkeypatch):
    a = _agent(monkeypatch)
    a._reward_rejected = 4
    assert a.phase2_stats()["reward_rejected"] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-hardening && uv run pytest tests/causal/test_agent_reward_hardening.py -v`
Expected: FAIL — `KeyError: 'reward_rejected'` / `AttributeError: _reward_rejected` / prompt sem "GRADUADO".

- [ ] **Step 3: Add the import**

In `agents/causal/agent.py`, line 25, change:

```python
from .goals import compile_reward, static_reward_check, goal_fn_from_reward, value_fn_from_reward
```

to:

```python
from .goals import (compile_reward, static_reward_check, goal_fn_from_reward,
                    value_fn_from_reward, accept_reward)
```

- [ ] **Step 4: Init `_reward_rejected`**

In `_init_causal_state`, right after `self._reward_src = None` (line 96), add:

```python
        self._reward_rejected = 0     # diag: rewards barradas pelo check comportamental
```

- [ ] **Step 5: Make `_build_reward_prompt` strict**

Replace `_build_reward_prompt` (lines 407-412) with:

```python
    def _build_reward_prompt(self, scene) -> str:
        objs = ", ".join(f"(color={o.color},size={o.size})" for o in scene.objects[:8])
        return ("Infira reward_function(state) que retorna (reward, goal_flag). REGRAS: "
                "(1) reward é um número GRADUADO — maior = mais perto de resolver, NÃO use só 0/1; "
                "(2) NÃO hardcode tamanhos/posições exatos (magic numbers) — use relações/contagens; "
                "(3) goal_flag=True SÓ quando o nível está realmente resolvido (raro). "
                "Olhe SÓ o state (lista de (tipo,{x,y,color,size,...})). "
                f"OBJETOS atuais: {objs}. "
                'Responda SÓ JSON {"type":"code","source":"def reward_function(state): ..."}')
```

- [ ] **Step 6: Rewrite `_try_learn_reward` and add the repair prompt**

Replace `_try_learn_reward` (lines 414-430) with:

```python
    def _try_learn_reward(self, scene) -> bool:
        """A: sintetiza a reward via LLM e ACEITA a 1ª que passa o check estático
        (anti-trapaça) E o comportamental (accept_reward: sem constante/sempre-True em
        estados reais). Self-repair com o motivo da rejeição até CAUSAL_REPAIR vezes."""
        if self._reward_fn is not None:
            return False
        states = [[(o.shape_hash, _obj_state(o)) for o in sc.objects]
                  for (sc, _k, _e) in self._buffer]
        states.append([(o.shape_hash, _obj_state(o)) for o in scene.objects])
        prompt = self._build_reward_prompt(scene)
        for _ in range(self._repair_max + 1):
            self._llm_calls += 1
            resps = (self._llm.complete_many(prompt, self._n_samples)
                     if self._n_samples > 1 else [self._llm.complete(prompt)])
            last_err = "sem resposta"
            for r in resps:
                g = parse_goal(r)
                src = g.get("source") if g and g.get("type") == "code" else None
                if not src or not static_reward_check(src):
                    last_err = "não passa no check estático (usa o state? sem global?)"
                    continue
                ok, reason = accept_reward(src, states)
                if ok:
                    self._reward_fn = compile_reward(src)
                    self._reward_src = src
                    return True
                self._reward_rejected += 1
                last_err = reason
            prompt = self._build_reward_repair_prompt(scene, last_err)
        return False

    def _build_reward_repair_prompt(self, scene, err) -> str:
        base = self._build_reward_prompt(scene)
        return base + f"\nA tentativa anterior FOI REJEITADA: {err}. Corrija e responda SÓ o JSON."
```

- [ ] **Step 7: Add the phase2_stats key**

In `phase2_stats`, after the `"reward_src": self._reward_src,` line, insert:

```python
            "reward_rejected": self._reward_rejected,
```

- [ ] **Step 8: Run the new tests to verify they pass**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-hardening && uv run pytest tests/causal/test_agent_reward_hardening.py -v`
Expected: PASS (5 tests).

- [ ] **Step 9: Run the full suite to confirm no regression**

Run: `cd /home/lkenzo/projetos/safe/ARC-AGI-3-Agents/.claude/worktrees/reward-hardening && uv run pytest tests/causal tests/kaggle -q`
Expected: PASS. Note: se algum teste antigo de `_try_learn_reward` assumia aceitação sem buffer, ele ainda passa pelo cold-start (`< min_states` → aceita); se algum popular buffer com reward binária esperando aceite, atualize-o para a intenção real (reward graduada) — reporte se acontecer.

- [ ] **Step 10: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_reward_hardening.py
git commit -m "feat: síntese de reward estrita (accept_reward + prompt graduado + self-repair + diag)"
```

---

## Notes for the offline notebook

`build_offline_notebook.py` embute `goals.py`/`agent.py` verbatim → o aperto flui pro run offline. Após o merge, regenerar com `uv run python kaggle/build_offline_notebook.py`. Validação real (multi-jogo via `OFFLINE_GAMES`): observar `reward_rejected > 0` nos jogos de reward ruim e se `reward_src` passa a ser graduada; se o LLM não conseguir, rejeita tudo (honesto) → próximo lever = Tycho.
