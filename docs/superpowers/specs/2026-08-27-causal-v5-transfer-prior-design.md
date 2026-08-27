# CausalObjectAgent v5 — Reuso inter-jogos: TransferPrior abstrato · Design

> Passo 4a da Fase 5 (a aposta central: ninguém no leaderboard reusa
> conhecimento entre jogos). Hoje cada instância do agente nasce fresca por
> jogo (`_init_causal_state` cria `CausalModel`/`NoveltyModel` novos) → todo
> aprendizado é descartado entre jogos.

## Duas realidades do harness que moldam o design

1. **Execução paralela.** `Swarm.main` cria uma instância de agente por jogo e
   roda todas em **threads paralelas no mesmo processo** (`Thread(target=a.main)`).
   Não há ordem "jogo A termina → B carrega". Reuso intra-run = um **objeto de
   conhecimento compartilhado entre as threads** (com lock).
2. **Transfer negativo.** A eval usa jogos **não vistos** e a semântica de ação
   difere por jogo (`ACTION1` faz coisas diferentes; `ACTION6@cell=2,3` é
   game/posição-específico). Persistir modelo cru arrisca transfer negativo. Só
   **modalidade de interação** generaliza.

## Decisões aprovadas

- **Escopo = 4a:** `TransferPrior` abstrato + compartilhamento intra-run
  (objeto singleton entre threads). **Sem disco** — persistência em disco e
  pré-treino offline pro Kaggle ficam para o 4b.
- **Abstração = features de ação/contexto** (generalizáveis), não modelos crus.

## Objetivo

Dar ao agente um **warm-start generalizável**: começar um jogo novo já
enviesado pras modalidades de interação historicamente produtivas, agregando
experiência de todos os jogos do run. Sem LLM/GPU, numpy/stdlib puro,
Kaggle-submittable.

## Arquitetura

Novo módulo `agents/causal/transfer.py` + termo opcional em
`agents/causal/policy.py` (`score`/`decide` ganham `prior=None`) e wiring em
`agents/causal/agent.py`. `perception.py`, `hud.py`, `causal_model.py`,
`novelty.py`, `instrumentation.py` **não mudam**.

### 1. Feature abstrata (`transfer.py`)

`abstract_feature(cand) -> str` — 3 buckets, nunca game-específicos:

```python
def abstract_feature(cand) -> str:
    if not cand.action.is_complex():
        return "simple"
    return "click_on_object" if cand.has_object else "click_empty"
```

- `"simple"` — qualquer ação de botão (agrupadas; o nome específico NÃO entra,
  pois sua semântica muda por jogo).
- `"click_on_object"` / `"click_empty"` — clique (`ACTION6`) sobre uma célula
  com/sem objeto. "Clicar em objeto tende a causar efeito" é uma verdade
  cross-game.

### 2. `TransferPrior` (`transfer.py`, thread-safe, serializável)

```python
import threading

W_PRIOR = 1.0
NEUTRAL_PRODUCTIVITY = 0.5


class TransferPrior:
    def __init__(self):
        self._counts = {}          # feature -> [n_produtivo, n_total]
        self._lock = threading.Lock()

    def observe(self, feature, effect_kind):
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

    def to_dict(self): ...          # {"counts": {feature: [np, nt]}}
    @classmethod
    def from_dict(cls, d): ...      # (usado no 4b; já deixamos pronto)
```

- `productivity(feature)` = P(a modalidade causa efeito ≠ `none`). Feature sem
  dados → `NEUTRAL_PRODUCTIVITY` (0.5): termo constante entre candidatos
  inéditos → não altera ranking até haver dados.
- `to_dict`/`from_dict` existem para o 4b (disco); aqui só validamos o
  roundtrip.

### 3. Singleton compartilhado (intra-run)

```python
_SHARED = TransferPrior()

def shared_prior() -> TransferPrior:
    return _SHARED

def reset_shared_prior() -> None:   # usado nos testes p/ isolar
    global _SHARED
    _SHARED = TransferPrior()
```

Todas as threads (agentes do `Swarm`) acessam a MESMA instância → cruzam
conhecimento durante o run, sem tocar no `Swarm`. Updates são protegidos pelo
lock interno do `TransferPrior`.

### 4. Termo na policy (`policy.py`)

`Policy.score(self, cand, model, seen_effects, budget_frac, novelty=None, prior=None)`
e `Policy.decide(..., prior=None)` ganham o parâmetro opcional. Com `prior=None`
o comportamento é idêntico ao v4 (mantém os 82 testes). Com `prior` presente:

```python
        if prior is not None:
            s += W_PRIOR * prior.productivity(abstract_feature(cand))
```

adicionado ao final do `score` (antes do `return s`). `W_PRIOR=1.0`:
`productivity ∈ [0,1]` → nudge de até +1 (comparável ao `has_object` +0.5;
menor que o termo de novidade, até ~3). Enviesa os primeiros lances de um jogo
novo pras modalidades produtivas; à medida que o modelo game-específico aprende,
o sinal de novidade domina.

### 5. Wiring no `agent.py`

- `_init_causal_state`: `from .transfer import shared_prior, abstract_feature`;
  `self._prior = shared_prior()`; `self._last_feature = None`.
- **Não** resetar `self._prior` no branch de RESET (é compartilhado/persistente).
- Ao decidir: guardar `self._last_feature = abstract_feature(cand)` junto com
  `self._last_key`.
- No bloco de fecha-loop (após `actual` calculado), registrar no prior:

```python
            if self._last_feature is not None:
                self._prior.observe(self._last_feature, actual.kind)
```

- Passar `prior=self._prior` ao `self._policy.decide(...)`.

## Fluxo de dados

Ação anterior observada → `prior.observe(last_feature, actual.kind)` atualiza a
produtividade da modalidade (agregada entre jogos via singleton) →
`decide`/`score` somam `W·productivity(abstract_feature(cand))` a cada
candidato → a policy prefere modalidades produtivas, especialmente no
cold-start do jogo novo.

## Erros e casos de borda

- **Feature inédita:** `productivity` → 0.5 (neutro); termo constante entre
  candidatos → sem viés até haver dados.
- **Concorrência:** `observe`/`productivity` sob `threading.Lock`; seguro para
  as threads paralelas do `Swarm`.
- **Compat:** `prior=None` mantém o caminho v4; nenhum teste v1–v4 muda.
- **Isolamento de teste:** `reset_shared_prior()` zera o singleton entre testes
  (senão o estado vaza de um teste pro outro).
- **Determinismo:** dado o estado do prior, `score` é determinístico.

## Testes (TDD, `tests/causal/`)

1. **`abstract_feature`** (`test_transfer.py`): ação simples → `"simple"`;
   clique com `has_object=True` → `"click_on_object"`; `False` → `"click_empty"`.
2. **`TransferPrior.productivity`:** sem dados → 0.5; após `observe` com efeitos
   `moved`/`none`, reflete `n_produtivo/n_total`; `none`/`None` não contam como
   produtivos.
3. **thread-safety básico:** N `observe` concorrentes (threads) → `n_total`
   final correto (soma).
4. **roundtrip `to_dict`/`from_dict`.**
5. **singleton:** `shared_prior()` retorna a mesma instância; `reset_shared_prior`
   troca por uma nova.
6. **`score` com prior** (`test_policy_prior.py`): candidato com feature
   produtiva (prior alto) pontua mais que um com feature improdutiva;
   `prior=None` reproduz o v4.
7. **`decide` com prior:** aceita `prior=` e escolhe o candidato de maior score
   com o termo do prior.
8. **integração no agente** (`test_agent_prior.py`): agente usa `shared_prior()`;
   após passos, o prior compartilhado recebeu `observe`; RESET não zera o prior;
   dois agentes distintos compartilham a MESMA instância de prior.
9. **Regressão:** os 82 testes v1–v4 seguem verdes.

## Fora de escopo (4b e além)

- **Persistência em disco** do `TransferPrior` (load no init / save no cleanup).
- **Pré-treino offline** e shipping read-only do prior no notebook Kaggle.
- Warm-start dos modelos crus com guarda de transfer negativo; features
  abstratas mais ricas.

## Critério de pronto

- `TransferPrior` thread-safe e serializável; `abstract_feature` com 3 buckets;
  `score`/`decide` com termo sob `prior`; agente usa o singleton compartilhado e
  o alimenta; RESET não zera.
- 82 testes v1–v4 + novos verdes.
- Rodar ao vivo 2+ jogos no MESMO run (`--game=vc33,ls20` sem filtro, ou vários)
  e conferir que o `TransferPrior` compartilhado acumulou contagens de mais de
  um jogo (evidência de cross-pollination). Log em `analysis/out/v5live/`.
