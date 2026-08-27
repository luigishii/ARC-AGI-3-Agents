# Baseline v1 — CausalObjectAgent (2026-08-27)

> Task 9 (final) do plano `docs/superpowers/plans/2026-08-27-causal-object-agent.md`.
> Escopo desta nota foi **ajustado pelo controller** em relação ao brief original
> (`.superpowers/sdd/2026-08-27-causal-object-agent/task-9-brief.md`) por dois motivos
> factuais: (1) não há `ARC_API_KEY` disponível neste ambiente — o run ao vivo contra a
> API não é possível agora; (2) `analysis/replay.py` não existe neste worktree (é um
> arquivo untracked que só existe no checkout principal) — por isso as métricas causais
> foram implementadas como módulo standalone versionado (`analysis/causal_metrics.py`),
> não como extensão de `replay.py`.

## Status da suíte de testes

Comando:

```bash
cd ARC-AGI-3-Agents && uv run pytest tests/causal/ -v
```

Resultado: **34 passed** (32 pré-existentes das Tasks 1–8 + 2 novas de
`tests/causal/test_causal_metrics.py`). Nenhuma falha, nenhum skip.

## O que está VALIDADO (unit/integration tests, sem API)

- **Percepção objeto-cêntrica** (`tests/causal/test_perception.py`,
  `test_perception_match.py`, `test_effect.py`): segmentação por cor ignorando
  background, bbox/centroide, `parse` usa a última camada da pilha, matching de
  objeto por id estável entre frames (move/recolor mantêm id, objeto novo ganha id),
  e as 6 categorias de `Effect` (`none`, `moved`, `disappeared`, `appeared`,
  `recolored`, `structural`).
- **Modelo causal** (`tests/causal/test_causal_model.py`): `observe`/`predict` de
  efeito modal por `(ação, tipo-de-objeto)`, previsão para chave desconhecida,
  flag de progresso (`is_progress`), tracking de acurácia de previsão, e
  roundtrip de serialização (`to_dict`/`from_dict`).
- **Policy** (`tests/causal/test_policy_candidates.py`,
  `test_policy_decide.py`): geração de candidatos (ação simples = 1 candidato;
  ação complexa = 1 candidato por objeto ancorado, com fallback se não há
  objetos), score de 4 termos, preferência por ação não-tentada sobre
  "sabidamente nada", preferência por ação de progresso, determinismo com
  `epsilon=0`, aceitação de `available_actions` como ints.
- **Loop causal fechando end-to-end** (`tests/causal/test_agent_integration.py`,
  `test_agent_smoke.py`): `CausalObjectAgent` fecha o loop
  percepção→modelo→policy em 2 passos consecutivos, detecta level-up como
  progresso, RESET quando `NOT_PLAYED`/`GAME_OVER`/`full_reset`, retorna ação
  disponível durante o jogo.
- **Logging deferido capturando o efeito observado**
  (`tests/causal/test_instrumentation.py`,
  `test_agent_integration.py::test_deferred_log_records_observed_effect`):
  `Instrumentation.log` acumula em memória e opcionalmente grava JSONL; o agente
  guarda o registro pendente sem `actual` no passo em que decide a ação e só
  grava a linha completa (com `actual`) no passo seguinte, quando o efeito real
  já foi observado — confirmado que o `actual` gravado bate com o efeito
  observado no frame seguinte.
- **Métricas offline** (`tests/causal/test_causal_metrics.py`, novo):
  `summarize_causal_log` (n_actions, wasted, explore_vs_exploit, acurácia e
  cobertura finais) e `plot_causal_metrics` (gera PNG a partir de um JSONL
  sintético) sobre um log fabricado em `tmp_path` — sem chamadas à API.

## O que está PENDENTE (bloqueado)

**Run ao vivo contra a API e o gate empírico do spec §5** — bloqueado por falta de
`ARC_API_KEY` neste ambiente. O spec exige, para "v1 pronto":

> completar ≥1 nível num jogo que o random não completa **E** acurácia de previsão
> >70% em jogos com ≥20 passos explorados.

Nenhuma dessas duas condições foi medida. Não há `levels_completed`,
`ações-por-nível`, acurácia de previsão real, nem cobertura reais de um jogo —
os números usados nos testes de `causal_metrics.py` são sintéticos, só para
validar a mecânica de leitura/plot/summary.

Quando houver `ARC_API_KEY` (em `.env`), rodar:

```bash
CAUSAL_LOG=analysis/out/causal.jsonl uv run main.py --agent=causalobject --game=ls20
uv run python analysis/causal_metrics.py --causal-log analysis/out/causal.jsonl
```

O primeiro comando fia o `Instrumentation` do agente para gravar cada ação em
`analysis/out/causal.jsonl` (via `os.environ.get("CAUSAL_LOG")` lido em
`_init_causal_state`, `agents/causal/agent.py`) e roda até `WIN` ou esgotar
`MAX_ACTIONS=80`. O segundo lê esse JSONL, imprime o `summary` (n_actions,
`final_prediction_accuracy`, `final_coverage_keys`, `wasted`,
`explore_vs_exploit`) e salva `analysis/out/causal_metrics.png` com as curvas
de `prediction_accuracy` e `coverage_keys` por passo. Repetir para o agente
`random` no mesmo jogo para ter o comparativo pedido pelo gate ("nível que o
random não completa").

## Como interpretar as métricas

`prediction_accuracy` subindo e se estabilizando indica que o modelo causal
está convergindo para regras confiáveis por `(ação, tipo-de-objeto)`;
`coverage_keys` subindo mostra que a policy ainda está explorando o espaço
ação×objeto — quando as duas curvas estabilizam juntas (accuracy alta,
coverage platô), o agente esgotou o que há para aprender no jogo e deveria
estar em modo EXPLOIT quase puro; se `wasted` (ações com efeito "none") for
alto perto do fim do orçamento de ações, é sinal de que a policy está
desperdiçando passos em vez de convergir para ações de progresso.
