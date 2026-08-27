# Design — CausalObjectAgent v2: HUD-masking + matching por sobreposição

> Data: 2026-08-27 · Fase 5 (iteração) · Projeto ARC Prize 2026 / ARC-AGI-3
> Status: **design aprovado, pronto para plano de implementação**
> Base: v1 (`agents/causal/`, spec `2026-08-27-causal-object-agent-design.md`).

## 1. Motivação (grounded no run real)

O v1 rodou ao vivo (2026-08-27) e completou **0 níveis** em `ls20` (keyboard) e
`vc33` (click). A inspeção dos frames reais do `vc33` revelou o bloqueador:

- **Todas as mudanças em 80 cliques ocorreram na linha 0** (um HUD/contador no topo).
  O jogo real (slider, cursor, alvos) **nunca foi tocado**. Os cliques do agente não
  afetaram o estado do jogo — só fizeram o contador do HUD tiquetaquear 1–2 px.
- Nosso `compute_effect` classificou **~100% das transições como `structural`**,
  porque uma mudança de 1–2 px muda o `shape_hash` de um objeto → o matching estrito
  (que exige `shape_hash` idêntico) perde a identidade → o objeto "some" e um novo
  "aparece" → `structural` (≥2 mudanças).

Consequência: o modelo causal aprende **ruído de HUD**, não a dinâmica do jogo. Duas
correções de fundo destravam o aprendizado do sinal real:

1. **HUD-masking** por independência-de-ação (o HUD muda a cada ação).
2. **Identidade de objeto tolerante a mudança de forma** (matching por sobreposição/IoU).

Referência competitiva: a disciplina "hudmask" do agente *duck* (Tufa Labs) descobre
o HUD exatamente por ação-independência — confirmando esta direção.

## 2. Objetivo e não-objetivos

**Objetivo:** limpar o sinal do efeito para que o `CausalModel` aprenda dinâmica real
em vez de ruído de HUD. Uma ação que só mexe no HUD deve produzir efeito `none`.

**Payoff sem mudar a Policy:** com o efeito limpo, uma ação inútil vira `none` → a
**penalidade de estagnação (−2) que já existe na Policy v1** faz o agente evitá-la, e o
termo de ganho-de-informação o empurra para ações de efeito real. Ou seja, v2 **não
altera a Policy** — só melhora a percepção/efeito.

**Não-objetivos (YAGNI):** goal-model, reuso entre jogos, descoberta de alvo de clique,
mudanças na Policy. Ficam para depois (dependem deste sinal limpo).

## 3. Componentes

### 3.1 `HudMask` (novo — `agents/causal/hud.py`)

Rastreador online por célula da grade 64×64.

- Estado: `change_count: np.ndarray(64,64) int`, `total: int`.
- `update(prev_grid, curr_grid)`: `change_count += (prev != curr)`; `total += 1`.
- `mask() -> np.ndarray(64,64) bool`: `True` onde `change_count/total ≥ HUD_THRESHOLD`
  (0.7) **e** `total ≥ HUD_MIN_SAMPLES` (5). Antes de 5 amostras retorna máscara toda
  `False` (começa cego — não mascara nada até ter evidência).
- Serializável (`to_dict`/`from_dict`) — gancho para reuso entre jogos (Fase futura),
  não usado no v2.

### 3.2 Máscara aplicada na percepção

`perception.parse(frame, hud_mask=None) -> Scene`: quando `hud_mask` é dado, as células
mascaradas são tratadas como **fundo** (excluídas da segmentação). Objetos inteiramente
dentro do HUD nunca entram na `Scene`. Assinatura retrocompatível (`hud_mask=None` =
comportamento v1). O agente passa `self._hud.mask()` a cada passo.

### 3.3 `match_objects` por sobreposição (IoU) — `perception.py`

Substitui o matching de 2 tiers estritos por 3 tiers, do mais forte ao mais tolerante;
cada objeto de `prev` casa no máximo com um de `curr` (`used`):

1. **Exato:** mesma cor + mesmo `shape_hash` + menor distância de centroide (fast path).
2. **IoU mesma cor:** `IoU(cells_prev, cells_curr) ≥ IOU_THRESHOLD` (0.3) e mesma cor.
3. **IoU qualquer cor:** `IoU ≥ IOU_THRESHOLD` (captura recolor+reshape).

`IoU = |cells_prev ∩ cells_curr| / |cells_prev ∪ cells_curr|`. Assim uma barra que
cresce 1 célula ou um cursor que desloca mantêm o `id`. Sem casamento → id fresco.
`IOU_THRESHOLD=0.3` é conservador o bastante para não fundir objetos distintos que só
se tocam.

### 3.4 Nova categoria de efeito — `compute_effect` (`causal_model.py`)

Com o id preservado sob mudança de forma, adicionar o kind **`morphed`**: mesmo id,
centroide **não** moveu (dr=dc=0), cor igual, mas o conjunto de células mudou (tamanho/
forma diferente). Ordem de decisão por objeto: `moved` (centroide mudou) → `recolored`
(cor mudou, células iguais) → `morphed` (células mudaram sem mover) → senão nada. >1
mudança simultânea (fora do HUD) → `structural`. `kinds` passa a ser
{none, moved, appeared, disappeared, recolored, morphed, structural}.

## 4. Fluxo por passo (mudanças no `agent.py`)

```
FrameData → grid = frame[-1]
         → [se prev_grid] self._hud.update(prev_grid, grid)
         → scene = match_objects(prev_scene, parse(grid, hud_mask=self._hud.mask()))
         → [loop causal] observe(prev_scene, last_key, scene, level_up)  # efeito já limpo
         → decide → set_data se complexa → reasoning → log (deferido)
         → guarda prev_grid, prev_scene, last_key, ...
```

`_init_causal_state` instancia `self._hud = HudMask()`. O `HudMask` é resetado junto do
resto no RESET.

## 5. Validação

**Unit tests (numpy, sem API):**
- `HudMask`: célula que muda em toda transição vira mascarada após `HUD_MIN_SAMPLES`;
  célula que mudou 1× não é mascarada; máscara vazia antes do mínimo de amostras.
- `parse(frame, hud_mask)`: objeto dentro da máscara é excluído da `Scene`; fora é mantido.
- `match_objects` IoU: objeto que cresce 1 célula mantém id (IoU alto); dois objetos
  distintos e distantes não fundem; recolor+reshape mantém id via tier 3.
- `compute_effect`: `morphed` para redimensionamento in-place; `moved`/`recolored`/`none`
  inalterados; transição em que só o HUD mudou → `none`.
- **Regressão nos frames reais do vc33** (fixture com o par de frames 0→1 do recording):
  após o HUD ser aprendido (alimentar o `HudMask` com transições suficientes), o efeito
  do tick de HUD em (0,61-62) vira `none` — não `structural`.

**Integração local (com `ARC_API_KEY`):** re-rodar `vc33` e `ls20`; medir se a fração de
efeitos `structural` cai drasticamente e se o agente passa a registrar efeitos reais
(`moved`/`morphed`) fora do HUD. Métrica de sucesso do v2: **os efeitos deixam de ser
~100% structural** e ações HUD-only viram `none` (visível em `wasted` subindo e
`explore_vs_exploit`/cobertura mais sãos). Completar nível ainda pode não acontecer
(depende de goal-model/descoberta de clique — próximas fases), e isso é esperado.

## 6. Compatibilidade Kaggle (submissão — Fase 6)

O v2 **não adiciona dependências** — só numpy/stdlib. Roda offline na avaliação Kaggle
(sem internet). O agente segue carregável pelo fluxo de submissão já mapeado nos
notebooks sample:

- `pip install --no-index --find-links .../arc_agi_3_wheels arc-agi python-dotenv`;
- copiar `ARC-AGI-3-Agents` p/ pasta gravável; injetar/usar `agents/causal/`;
- **enxugar `agents/__init__.py`** para importar só o necessário (o `__init__` original
  importa langgraph/smolagents avidamente → quebra offline);
- `.env` apontando pro gateway; `python main.py --agent=causalobject`.

Nenhum `kagglehub.dataset_download` em runtime. Construir o notebook de submissão em si
é a Fase 6; o v2 apenas mantém o código submission-clean (sem novas deps, sem I/O de rede).

## 7. Arquivos afetados

```
agents/causal/hud.py           # novo: HudMask
agents/causal/perception.py    # parse(hud_mask=), match_objects por IoU
agents/causal/causal_model.py  # compute_effect: kind "morphed"
agents/causal/agent.py         # instanciar/atualizar HudMask no loop
tests/causal/test_hud.py       # novo
tests/causal/test_perception*.py, test_effect.py  # novos casos IoU/morphed/mask
```

## 8. Riscos

- **HUD_THRESHOLD/IOU_THRESHOLD mal calibrados:** thresholds fixos (0.7 / 0.3) podem
  mascarar gameplay que muda muito, ou não fundir objetos que deveriam. Mitigação:
  começar conservador; os unit tests + re-run medem; ajustar por evidência.
- **Objeto legítimo que muda a cada ação** (raro) seria mascarado como HUD. Aceitável no
  v2 (o HUD é de longe o caso comum); refino por correlação-com-clique fica para depois.
- **Bootstrap cego (5 primeiras ações sem máscara):** as primeiras transições poluem o
  modelo com ruído de HUD; com ≤80 ações o custo é pequeno e o modelo se corrige.
