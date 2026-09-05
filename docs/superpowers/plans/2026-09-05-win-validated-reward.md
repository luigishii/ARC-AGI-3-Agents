# Win-Validated Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Depois da 1ª vitória de um jogo, escolher entre as rewards grounded existentes a que *explica* a vitória (argmax no frame vencedor + gradiente positivo) e deixar a camada `rprog` dirigir o nível seguinte acima do `2phase`.

**Architecture:** Módulo puro novo `agents/causal/winselect.py` (avaliação de candidatos sobre a trajetória do nível). `agent.py` acumula os estados do nível corrente, chama a seleção no level-up, adota a reward escolhida em `_try_learn_reward` (antes do `template`) e, quando ela existe, consulta `_rprog_decide(uncapped=True)` antes do bloco `2phase`. Tudo sob `CAUSAL_WINREWARD=1` (default off, caminho off byte-idêntico).

**Tech Stack:** Python 3.12, numpy, pytest (`uv run pytest`). Spec: `docs/superpowers/specs/2026-09-05-win-validated-reward-design.md`.

## Global Constraints

- Flag `CAUSAL_WINREWARD` default **off**; com off nenhuma chamada nova acontece.
- Só seleciona rewards já existentes em `agents/causal/goals.py`; nenhuma reward nova.
- Estado = `[(shape_hash, _obj_state(o)) for o in scene.objects]` (mesmo contrato das rewards).
- Commits em PT-BR, prefixo `feat(winreward):`/`test(winreward):`, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_0178d4cZCTi9F56mek7hJJP1`.
- Rodar a suíte inteira antes do commit final: `uv run pytest tests/causal tests/kaggle tests/unit -q` (595 verdes na base).

---

### Task 1: `winselect.py` — pontuação e seleção

**Files:**
- Create: `agents/causal/winselect.py`
- Test: `tests/causal/test_winselect.py`

**Interfaces:**
- Produces: `explain_score(values: list[float], win_idx: int) -> tuple[bool, float]` e `select_win_reward(candidates: list[tuple[str, Callable]], level_states: list, win_state) -> tuple[str, Callable, float] | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/causal/test_winselect.py
import math
from agents.causal.winselect import explain_score, select_win_reward


def _states(n):
    # estados sinteticos: 1 objeto cuja coordenada x cresce com o tempo
    return [[("h", {"x": i, "y": 0, "color": 1, "shape": "h", "size": 4})] for i in range(n)]


def test_explain_score_argmax_and_rho():
    is_top, rho = explain_score([1.0, 2.0, 3.0, 4.0], win_idx=3)
    assert is_top and rho > 0.9


def test_explain_score_not_top():
    is_top, rho = explain_score([5.0, 2.0, 3.0, 4.0], win_idx=3)
    assert not is_top


def test_explain_score_constant_is_zero_rho():
    is_top, rho = explain_score([1.0, 1.0, 1.0], win_idx=2)
    assert is_top and rho == 0.0


def test_select_picks_increasing_argmax_candidate():
    st = _states(5); win = st[-1]; level = st[:-1]
    up = ("up", lambda s: (float(s[0][1]["x"]), False))          # cresce, win e argmax
    down = ("down", lambda s: (-float(s[0][1]["x"]), False))     # decresce
    const = ("const", lambda s: (1.0, False))                    # sem gradiente
    got = select_win_reward([down, const, up], level, win)
    assert got is not None and got[0] == "up" and got[2] > 0.9


def test_select_skips_exception_and_nonfinite():
    st = _states(5); win = st[-1]; level = st[:-1]
    boom = ("boom", lambda s: 1 / 0)
    nan = ("nan", lambda s: (math.nan, False))
    up = ("up", lambda s: (float(s[0][1]["x"]), False))
    assert select_win_reward([boom, nan, up], level, win)[0] == "up"


def test_select_none_when_no_candidate_explains():
    st = _states(5); win = st[-1]; level = st[:-1]
    down = ("down", lambda s: (-float(s[0][1]["x"]), False))
    assert select_win_reward([down], level, win) is None


def test_select_none_with_too_few_states():
    st = _states(3); win = st[-1]; level = st[:-1]       # 2 estados de nivel < 3
    up = ("up", lambda s: (float(s[0][1]["x"]), False))
    assert select_win_reward([up], level, win) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/causal/test_winselect.py -q`
Expected: FAIL com `ModuleNotFoundError: agents.causal.winselect`.

- [ ] **Step 3: Write the implementation**

```python
# agents/causal/winselect.py
"""Seleção da reward que EXPLICA a 1ª vitória de um jogo.

Dada a trajetória rotulada de um nível (estados decisão→decisão + estado vencedor),
uma reward candidata "explica" a vitória se (1) o estado vencedor é o de maior valor
(empate permitido) e (2) o valor cresce com o tempo (Spearman > 0). Entre as válidas,
vence a de maior Spearman. Puro numpy/stdlib; exception-safe por candidato.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

MIN_LEVEL_STATES = 3


def _spearman(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    v = np.asarray(values, dtype=float)
    t = np.arange(len(v), dtype=float)
    if np.std(v) == 0.0:
        return 0.0
    rv = np.argsort(np.argsort(v)).astype(float)
    rt = np.argsort(np.argsort(t)).astype(float)
    rho = float(np.corrcoef(rt, rv)[0, 1])
    return 0.0 if math.isnan(rho) else rho


def explain_score(values: list[float], win_idx: int) -> tuple[bool, float]:
    """(is_top, rho): o valor em win_idx é >= todos os outros? e Spearman(tempo, valor)."""
    if not values:
        return (False, 0.0)
    win = values[win_idx]
    is_top = all(win >= v for v in values)
    return (is_top, _spearman(values))


def _value(fn: Callable, state) -> float | None:
    try:
        r = fn(state)
    except Exception:
        return None
    v = r[0] if isinstance(r, (tuple, list)) else r
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def select_win_reward(candidates: list[tuple[str, Callable]], level_states: list,
                      win_state) -> tuple[str, Callable, float] | None:
    """Melhor candidato que explica a vitória, ou None (sem evidência/ninguém explica)."""
    if len(level_states) < MIN_LEVEL_STATES:
        return None
    traj = list(level_states) + [win_state]
    best = None
    for name, fn in candidates:
        vals = []
        for st in traj:
            v = _value(fn, st)
            if v is None:
                vals = None
                break
            vals.append(v)
        if not vals or len({round(v, 6) for v in vals}) < 2:
            continue
        is_top, rho = explain_score(vals, len(vals) - 1)
        if not is_top or rho <= 0.0:
            continue
        if best is None or rho > best[2]:
            best = (name, fn, rho)
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/causal/test_winselect.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/causal/winselect.py tests/causal/test_winselect.py
git commit -m "feat(winreward): winselect — reward que explica a 1a vitoria (argmax no win + Spearman>0)"
```

---

### Task 2: `agent.py` — acumular estados do nível e selecionar no level-up

**Files:**
- Modify: `agents/causal/agent.py` (`_init_causal_state`; close-loop perto de `self._buffer.append((self._prev_scene, self._last_key, actual.kind))`; bloco `if level_up:` onde já existe `win_scene = parse(wg, hud_mask=self._hud.mask())`; `_try_learn_reward` ramo grounded antes de `if self._win_template is not None:`; `phase2_stats`)
- Test: `tests/causal/test_agent_winreward.py`

**Interfaces:**
- Consumes: `select_win_reward` (Task 1); `grounded_*_reward_fn` de `goals.py`; `_obj_state`, `_pick_target`, `_background_colors` já em `agent.py`.
- Produces: atributos `self._winreward_on: bool`, `self._level_states: list`, `self._win_reward: tuple[str, Callable, float] | None`; método `_win_reward_candidates(win_scene) -> list[tuple[str, Callable]]`; `phase2_stats` ganha `win_reward` e `win_rho`.

- [ ] **Step 1: Write the failing test**

```python
# tests/causal/test_agent_winreward.py
"""Level-up com CAUSAL_WINREWARD: a reward do proximo nivel e a candidata que EXPLICA a
vitoria (aqui multi-align: 2 objetos da mesma cor convergem ate encostar), nao o
template(win-grid). Sem o flag, comportamento antigo (template)."""
from arcengine import GameAction, GameState

from agents.causal.agent import CausalObjectAgent

CLICK = [GameAction.ACTION6]


class _Frame:
    def __init__(self, frame, levels=0):
        self.frame = frame
        self.state = GameState.NOT_FINISHED
        self.levels_completed = levels
        self.available_actions = CLICK
        self.full_reset = False


def _layer(gap):
    """Fundo 0; bloco A cor 5 fixo em x=10..12; bloco B cor 5 a `gap` colunas a direita."""
    g = [[0] * 64 for _ in range(64)]
    for r in range(30, 33):
        for c in range(10, 13):
            g[r][c] = 5
        for c in range(13 + gap, 16 + gap):
            g[r][c] = 5
    return g


def _agent(monkeypatch, **env):
    env.setdefault("CAUSAL_LLM", "0")
    env.setdefault("CAUSAL_GK", "0")
    env.setdefault("CAUSAL_GROUNDED", "1")
    env.setdefault("CAUSAL_RPROG", "1")
    env.setdefault("CAUSAL_MAX_ACTIONS", "10000")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = CausalObjectAgent.__new__(CausalObjectAgent)
    a.action_counter = 0
    a.game_id = "zzzz"
    a.MAX_ACTIONS = 10000
    a._init_causal_state()
    return a


def _play_level(a):
    """Blocos convergem 20→...→2 (frames do nivel), depois o frame de VITORIA
    (pilha [camada vencedora gap=0, init do proximo nivel gap=20])."""
    for gap in (20, 16, 12, 8, 4, 2):
        a.choose_action([], _Frame([_layer(gap)]))
        a.action_counter += 1
    a.choose_action([], _Frame([_layer(0), _layer(20)], levels=1))
    a.action_counter += 1


def test_levelup_selects_reward_that_explains_win(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_WINREWARD="1")
    _play_level(a)
    assert a._win_reward is not None and a._win_reward[0] == "multi-align"
    assert a._win_reward[2] > 0.5
    a.choose_action([], _Frame([_layer(20)], levels=1))       # 1o passo do L1
    assert a._reward_src.startswith("win:multi-align")
    st = a.phase2_stats()
    assert st["win_reward"] == "multi-align" and st["win_rho"] > 0.5


def test_levelup_without_flag_keeps_template(monkeypatch):
    a = _agent(monkeypatch)
    _play_level(a)
    assert a._win_reward is None
    a.choose_action([], _Frame([_layer(20)], levels=1))
    assert a._reward_src == "grounded:template(win-grid)"


def test_level_states_reset_on_levelup(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_WINREWARD="1")
    _play_level(a)
    assert a._level_states == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/causal/test_agent_winreward.py -q`
Expected: FAIL com `AttributeError: ... '_win_reward'`.

- [ ] **Step 3: Implement in `agent.py`**

(a) Import, junto dos outros `from .` no topo:

```python
from .winselect import select_win_reward
```

(b) Em `_init_causal_state`, perto de `self._grounded = os.environ.get("CAUSAL_GROUNDED", "0") != "0"`:

```python
        # Lever "reward validada pela vitoria" (spec 2026-09-05): apos o 1o level-up,
        # escolhe entre as rewards grounded a que EXPLICA a vitoria e deixa o rprog
        # dirigir acima do 2phase. Default off (caminho off byte-identico).
        self._winreward_on = os.environ.get("CAUSAL_WINREWARD", "0") != "0"
        self._level_states: list = []      # estados decisao->decisao do nivel corrente
        self._win_reward = None            # (nome, fn, rho) | None
```

(c) No close-loop, na linha imediatamente após `self._buffer.append((self._prev_scene, self._last_key, actual.kind))` (ramo decisão→decisão, `else` do `if level_up`):

```python
                if self._winreward_on and len(self._level_states) < 400:
                    self._level_states.append(
                        [(o.shape_hash, _obj_state(o)) for o in self._prev_scene.objects])
```

(d) No bloco `if level_up:`, logo após `win_scene = parse(wg, hud_mask=self._hud.mask())`:

```python
                if self._winreward_on:
                    win_state = [(o.shape_hash, _obj_state(o)) for o in win_scene.objects]
                    self._win_reward = select_win_reward(
                        self._win_reward_candidates(win_scene), self._level_states, win_state)
                    self._level_states = []
```

(e) Método novo, ao lado de `_adopt_grounded`:

```python
    def _win_reward_candidates(self, win_scene) -> list:
        """Rewards grounded candidatas a explicar a vitoria (nomes estaveis p/ telemetria)."""
        cands = [("multi-align", grounded_multi_reward_fn()),
                 ("pattern", grounded_pattern_reward_fn()),
                 ("pair", grounded_pair_reward_fn()),
                 ("count", grounded_count_reward_fn()),
                 ("diversity", grounded_diversity_reward_fn())]
        aid = self._move.avatar_id()
        objs = list(win_scene.objects)
        avatar_idx = next((i for i, o in enumerate(objs) if o.id == aid), None)
        if avatar_idx is not None:
            bg = _background_colors(win_scene)
            tidx = _pick_target(objs, avatar_idx, limit=None,
                                exclude_ids=self._move.avatar_parts(), bg_colors=bg)
            if tidx is not None:
                av, tg = objs[avatar_idx], objs[tidx]
                cands.append(("nav", grounded_reward_fn(av.color, tg.color,
                                                        avatar_size=av.size,
                                                        target_size=tg.size)))
        return cands
```

(f) Em `_try_learn_reward`, ramo grounded, **antes** de `if self._win_template is not None:`:

```python
            if self._win_reward is not None:
                name, fn, rho = self._win_reward
                self._reward_fn, self._reward_src = fn, f"win:{name}(rho={rho:.2f})"
                return True
```

(g) Em `phase2_stats`, junto de `"reward_src": self._reward_src,`:

```python
            "win_reward": None if self._win_reward is None else self._win_reward[0],
            "win_rho": None if self._win_reward is None else round(self._win_reward[2], 3),
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/causal/test_agent_winreward.py tests/causal/test_blind_fixes.py tests/causal/test_agent_reward_blind_nav.py -q`
Expected: todos passam. Se `test_levelup_selects_reward_that_explains_win` falhar por `_win_reward is None`, verificar que `_level_states` acumulou ≥3 estados (a 1ª chamada de `choose_action` não tem `_prev_scene`; com 6 frames sobram 5 transições).

- [ ] **Step 5: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_winreward.py
git commit -m "feat(winreward): level-up seleciona a reward que explica a vitoria e a adota no nivel seguinte (CAUSAL_WINREWARD)"
```

---

### Task 3: `agent.py` — `rprog` acima do `2phase` quando há reward validada

**Files:**
- Modify: `agents/causal/agent.py` (`_rprog_decide`; pilha em `choose_action`, antes do comentário `# (2b) two-phase:`)
- Test: `tests/causal/test_agent_winreward.py` (acrescentar)

**Interfaces:**
- Consumes: `self._win_reward`, `self._rprog` (dict key → deque de Δ), `self._rprog_on`.
- Produces: `_rprog_decide(self, cands, uncapped: bool = False) -> str | None`.

- [ ] **Step 1: Write the failing tests** (acrescentar ao arquivo da Task 2)

```python
from collections import deque


def _prime_rprog(a, key, deltas):
    a._rprog[key] = deque(deltas, maxlen=10)


def test_rprog_precedes_two_phase_with_win_reward(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_WINREWARD="1")
    _play_level(a)                                        # define _win_reward
    a.choose_action([], _Frame([_layer(20)], levels=1))   # adota a reward win:
    a.action_counter += 1
    a._last_effect_kind = "structural"                    # o 2phase dispararia
    a.choose_action([], _Frame([_layer(16)], levels=1))
    key = a._pending_log["reasoning"]["key"]              # chave presente neste layout
    _prime_rprog(a, key, [0.5, 0.6, 0.7])
    a._last_effect_kind = "structural"
    a.action_counter += 1
    a.choose_action([], _Frame([_layer(16)], levels=1))   # mesmo layout -> mesma chave existe
    assert a._pending_log["reasoning"]["layer"] == "rprog"
    assert a._pending_log["reasoning"]["key"] == key


def test_two_phase_still_first_without_flag(monkeypatch):
    a = _agent(monkeypatch)                                # flag off
    _play_level(a)
    a.choose_action([], _Frame([_layer(20)], levels=1))
    a.action_counter += 1
    a._last_effect_kind = "structural"
    a.choose_action([], _Frame([_layer(16)], levels=1))
    key = a._pending_log["reasoning"]["key"]
    _prime_rprog(a, key, [0.5, 0.6, 0.7])
    a._last_effect_kind = "structural"
    a.action_counter += 1
    a.choose_action([], _Frame([_layer(16)], levels=1))
    assert a._pending_log["reasoning"]["layer"] != "rprog"


def test_rprog_uncapped_ignores_max(monkeypatch):
    a = _agent(monkeypatch, CAUSAL_WINREWARD="1")
    a._rprog_fires = a._rprog_max

    class C:  # candidato minimo
        key = "ACTION6@c5s0@1,1"
    _prime_rprog(a, C.key, [0.5, 0.6, 0.7])
    assert a._rprog_decide([C()]) is None
    assert a._rprog_decide([C()], uncapped=True) == C.key
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/causal/test_agent_winreward.py -q`
Expected: `test_rprog_uncapped_ignores_max` FAIL com `TypeError: unexpected keyword 'uncapped'`; `test_rprog_precedes_two_phase_with_win_reward` FAIL com layer `2phase`.

- [ ] **Step 3: Implement**

(a) `_rprog_decide`:

```python
    def _rprog_decide(self, cands, uncapped: bool = False):
        """B′: escolhe a ação de maior Δ médio de reward real (>0). Min 3 observações
        por key, cap de _rprog_max fires por episódio (evita monopolizar) — salvo
        `uncapped` (reward validada pela vitória: o rprog É o driver). Sem
        dados/positivo/cap estourado → None."""
        if not uncapped and self._rprog_fires >= self._rprog_max:
            return None
```
(resto do método inalterado)

(b) Na pilha de `choose_action`, imediatamente **antes** do comentário `# (2b) two-phase: se a ultima acao teve efeito, tenta a proxima fase.`:

```python
        # (2a) reward VALIDADA pela vitoria: o rprog (sobe a reward real) dirige ANTES
        # das heuristicas 2phase — em jogo click-only onde todo botao tem efeito o
        # 2phase disparava todo passo (vc33: 144/200 acoes em round-robin) e o rprog
        # nunca decidia. Sem sinal (Δ medio > 0 com >=3 obs) cai no resto da pilha.
        if (cand is None and self._winreward_on and self._win_reward is not None
                and self._rprog_on and cands):
            rk = self._rprog_decide(cands, uncapped=True)
            if rk is not None:
                cand = keymap.get(rk)
                layer = layer or "rprog"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/causal/test_agent_winreward.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/causal/agent.py tests/causal/test_agent_winreward.py
git commit -m "feat(winreward): rprog (sem cap) decide acima do 2phase quando a reward foi validada pela vitoria"
```

---

### Task 4: Flag nos builders + notebooks + suíte completa

**Files:**
- Modify: `kaggle/build_notebook.py` (linha `"CAUSAL_GROUNDED=1\n"`), `kaggle/build_offline_notebook.py` (linha `"CAUSAL_GROUNDED=1\n"`)
- Test: `tests/kaggle/test_build_notebook.py`, `tests/kaggle/test_build_offline_notebook.py` (acrescentar 1 teste cada, copiando o teste existente que verifica `CAUSAL_GROUNDED=1` no `.env` gerado e trocando por `CAUSAL_WINREWARD=1`)

- [ ] **Step 1: Write the failing tests** (duplicar o teste existente de `CAUSAL_GROUNDED=1` em cada arquivo, com `CAUSAL_WINREWARD=1`)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/kaggle -q`
Expected: os 2 novos falham.

- [ ] **Step 3: Implement** — adicionar após a linha `"CAUSAL_GROUNDED=1\n"` em ambos os builders:

```python
    "CAUSAL_WINREWARD=1\n"      # reward validada pela 1a vitoria + rprog acima do 2phase
```

- [ ] **Step 4: Regenerate notebooks and run the full suite**

```bash
uv run python kaggle/build_notebook.py
uv run python kaggle/build_offline_notebook.py
uv run pytest tests/causal tests/kaggle tests/unit -q
```
Expected: 595 + novos passam; `git status` mostra os 2 `.ipynb` modificados.

- [ ] **Step 5: Commit**

```bash
git add kaggle/ tests/kaggle/
git commit -m "feat(winreward): CAUSAL_WINREWARD=1 nos builders; notebooks regenerados"
```

---

### Task 5: Validação real (loop local grátis)

- [ ] **Step 1:** com o baseline dos 25 encerrado (não rodar em paralelo: 429), rodar os 4 jogos com L1:

```bash
export CAUSAL_LLM=0 CAUSAL_GK=0 CAUSAL_CLICKMAP=1 CAUSAL_GROUNDED=1 CAUSAL_COVER=1 CAUSAL_FIX=1 \
       CAUSAL_RPROG=1 CAUSAL_IW=1 CAUSAL_ETA=1 CAUSAL_TYPED=1 CAUSAL_MAX_ACTIONS=200 CAUSAL_WINREWARD=1
for g in vc33 tn36 lp85 lf52; do timeout 900 uv run main.py --agent=causalobject --game=$g > /tmp/wr_$g.log 2>&1; grep -oE "levels completed [0-9]+" /tmp/wr_$g.log | tail -1; grep -oE "'win_reward': [^,]*, 'win_rho': [^,]*|'rprog_fires': [0-9]+|'layers': \{[^}]*\}" /tmp/wr_$g.log | tail -3; done
```

- [ ] **Step 2:** ler `win_reward` (bateu com o probe: multi-align em vc33/tn36?), `layers['rprog']` (dirigiu?), `levels_completed` (L2?). Registrar no CLAUDE.md.
