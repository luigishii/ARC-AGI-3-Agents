# ARC-AGI-3 Agents — Projeto de Competicao

## O que e
Agente autonomo para a competicao ARC-AGI-3 (Kaggle). O agente joga 25 mini-jogos (puzzles visuais em grid 64x64), tomando acoes (teclado 1-4, clique ACTION6, validar ACTION5, undo ACTION7) a cada frame. Meta: completar o maximo de niveis em 200 acoes por jogo.

## Ambiente de execucao
- **Kaggle Offline** (GPU RTX Pro 6000, sem internet)
- `ENVIRONMENTS_DIR = /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files` (25 metadata.json)
- `GPT_OSS_KERNEL_DIR = /kaggle/working/tkrepo/build/torch-universal` (Qwen3-32B local)
- `OPERATION_MODE=offline`, `CAUSAL_LLM=1`, `CAUSAL_MAX_ACTIONS=200`
- Model path: `/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1`
- Kernels dataset: `/kaggle/input/datasets/luigiishii/gpt-oss-offline-kernels`

## Arquitetura do agente

### Agente principal: `CausalObjectAgent` (`agents/causal/agent.py`)
Pipeline por frame:
1. **Percepcao** (`perception.py`, `perception_strategy.py`): grid 64x64 -> objetos (cor, centroid, bbox, size, shape_hash)
2. **HUD mask** (`hud.py`): detecta e mascara barras de budget (linha 0/63, coluna 62/63)
3. **Modelo causal** (`causal_model.py`): aprende regras acao->efeito em objetos
4. **Ontologia tipada** (`typed_model.py`, `ontology.py`): world-model fatorado por tipo de objeto (f_tau)
5. **Reward grounded** (`goals.py`): reward sintetizada espacialmente (manhattan avatar->alvo, alinhamento multi-obj, pattern match)
6. **Politica** (`policy.py`, `ranker.py`): selecao de acao via epsilon-greedy + novelty + reward progress
7. **IW planning** (`iw.py`): busca width-bounded goal-directed
8. **Navegacao** (`navigate.py`): MovementModel aprende direcao de cada tecla
9. **LLM** (`llm.py`): Qwen3-32B local para sintese de reward e raciocinio direto passo-a-passo
10. **Clickmap** causal: chave de clique por (cor, tamanho) — evita clicar HUD/fundo/parede

### Flags de controle (env vars)
| Flag | Default | O que faz |
|------|---------|-----------|
| `CAUSAL_LLM` | 0 | Ativa LLM (Qwen3-32B) para sintese de reward e raciocinio |
| `CAUSAL_DIRECT` | 0 | Raciocinio direto passo-a-passo (score-max Lever #2) |
| `CAUSAL_TYPED` | 0 | World-model tipado por objeto |
| `CAUSAL_ETA` | 0 | Exploracao por erro de ontologia |
| `CAUSAL_IW` | 0 | IW planning goal-directed |
| `CAUSAL_RPROG` | 0 | Progresso model-free por reward real |
| `CAUSAL_COVER` | 0 | Exploracao por cobertura + anti-fixacao |
| `CAUSAL_CLICKMAP` | 0 | Clickmap causal (cor,tamanho) |
| `CAUSAL_GROUNDED` | 0 | Reward grounded (manhattan, alinhamento, pattern) |
| `CAUSAL_FIX` | 0 | Guarda global anti-fixacao |
| `CAUSAL_MAX_ACTIONS` | 80 | Limite de acoes por jogo |
| `CAUSAL_EFFORT` | medium | Esforco de raciocinio LLM (low/medium/high) |
| `OFFLINE_GAMES` | "" | Filtro de jogos (prefixos separados por virgula; "" = todos) |

## Taxonomia dos 25 jogos (6 classes)
Ver `docs/GAMES.md` para detalhes completos.
- **A. Navegacao avatar->alvo** (teclado): dc22, g50t, tu93, sc25, ls20, m0r0
- **B. Sokoban / empurrar-blocos**: ka59, wa30, su15
- **C. Manipulacao/posicionamento de pecas** (selec+mover+girar): vc33, ar25, cn04, r11l, s5i5, lf52, lp85
- **D. Preenchimento/pintura por padrao**: cd82, ft09, re86
- **E. Sequencia/ordenacao + VALIDAR**: sb26, tr87, sk48, tn36
- **F. Roteamento de fluxo**: sp80
- **Novo (nao documentado)**: bp35

## Jogos jogaveis confirmados (25)
sk48, tn36, m0r0, bp35, cn04, dc22, tu93, lp85, ka59, wa30, vc33, lf52, r11l, sc25, sp80, ar25, sb26, cd82, re86, s5i5, ls20, ft09, su15, tr87, g50t

## Problemas conhecidos (diagnosticados)
1. **Agente colapsa em ACTION6 (clique cego)** — ignora teclado (1-4), undo (7), validar (5). Exemplo: vc33 fez 200 ACTION6 sem completar nenhum nivel (L0 baseline = 7 acoes).
2. ~~**Clica HUD/fundo/parede**~~ — FIX: size penalty (-4.0 para obj_size>100) em Policy.score(). CAUSAL_CLICKMAP ajuda mas nao bastava sozinho.
3. **Reward por contagem-de-cor e estruturalmente incompativel** — vitoria e sempre espacial/posicional. Fix parcial: CAUSAL_GROUNDED.
4. **LLM muito lento (~30s/acao)** — 0.04 fps. Gargalo e a inferencia do Qwen3-32B no Kaggle.
5. ~~**Rprog monopoliza decisoes**~~ — FIX: sliding window (deque maxlen=10), min 3 obs, cap 20 fires. Rprog bonus integrado no Policy.score().
6. **Puzzles L0 sao curtos (2-13 acoes)** mas o agente desperdiça 200 sem resolver.

## Estrutura de diretorios
```
agents/
  causal/           # agente principal (CausalObjectAgent)
    agent.py        # loop principal
    perception.py   # grid -> objetos
    causal_model.py # regras acao->efeito
    goals.py        # reward sintetizada + grounded
    llm.py          # interface LLM (Qwen3-32B local ou API)
    policy.py       # selecao de acao
    iw.py           # IW planning
    navigate.py     # MovementModel
    hud.py          # deteccao/mascaramento de HUD
    ...
  swarm.py          # orquestracao multi-jogo
  recorder.py       # gravacao de replays
kaggle/
  build_notebook.py          # gera notebook de submissao online
  build_offline_notebook.py  # gera notebook offline (de-blinding)
  build_eval_notebook.py     # gera notebook de avaliacao
docs/
  GAMES.md                   # enciclopedia dos 25 jogos
  superpowers/plans/         # planos de melhoria
  superpowers/specs/         # specs de design
tests/causal/                # testes unitarios e de integracao
analysis/                    # metricas e analise de rollouts
```

## Como rodar

### Local (online, com API key)
```bash
uv run main.py --agent=causalobject --game=vc33
```

### Kaggle offline (sem internet)
O notebook e gerado por `kaggle/build_offline_notebook.py`.
Env vars chave: `OPERATION_MODE=offline`, `OFFLINE_GAMES=vc33,ls20`, `CAUSAL_LLM=1`.
O modelo Qwen3-32B carrega 1x e joga todos os jogos no mesmo processo.

### Testes
```bash
pytest tests/causal/ -x -q
```

## Convencoes
- Codigo e comentarios em portugues (pt-BR)
- Commit messages em ingles, prefixo convencional (feat/fix/docs/test)
- Linter: ruff
- Python >= 3.12
- Coordenadas: x=col, y=row (convencao do arcengine)
