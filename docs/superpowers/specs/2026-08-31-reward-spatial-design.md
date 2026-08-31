# Reward espacial — design

**Data:** 2026-08-31
**Status:** spec aprovada (escopo: posição + grid célula-a-célula)

## Problema (provado com dados)

Score travado em **0.08** (empate no ruído com 0.09) apesar de guarda/cobertura/hardening
validados ao vivo. Atribuição limpa: **100% do gargalo restante é goal-discovery** — o agente
não sabe qual é a meta pra persegui-la.

Decodificação de 3 jogos (`analysis/replay.py` + logs) provou que **a meta de todos é espacial**:
- **ls20** (teclado): avatar (bloco 3×5) anda 5px/tecla rumo a alvo fixo; contagem de cor do
  avatar é constante (45px) — só a **posição** muda.
- **ka59** (alinhamento): cursor move rumo a quadrado amarelo estático; **contagens de cor
  LITERALMENTE constantes o jogo todo** → `color_counts>=N` é matematicamente incapaz de mudar.
- **sc25** (grid 3×3 tipo Lights-Out): meta = casar padrão **célula-a-célula**; contagem
  agregada não distingue resolvido de não-resolvido.

**Causa raiz isolada:** o `state` que a `reward_function` recebe **já tem posição**
(`_obj_state`, `agent.py:36` → `{"x":col,"y":row,"color":..,"shape":..}`), MAS o prompt de
síntese `_build_reward_prompt` (`agent.py:451`) mostra ao LLM **só `(color=X, size=Y)` por
objeto — esconde x,y**. O Qwen escreve reward de contagem-de-cor porque é a única coisa que o
prompt exibe. A informação espacial existe no `state` mas é invisível pro modelo que escreve a
função. (Bug secundário: o prompt mostra `size`, que nem está no `state`.)

## Solução

Lever **majoritariamente prompt-engineering**: reescrever `_build_reward_prompt` pra expor a
estrutura espacial que o `state` já carrega, mais few-shot de rewards espaciais. Não precisa de
nova percepção — as células de um tabuleiro são objetos com x,y+color, então casamento
célula-a-célula é expressável a partir da mesma lista de objetos.

### Componente único: `_build_reward_prompt` (agents/causal/agent.py)

Reescrever pra incluir, na ordem:

1. **Objetos com posição** — por objeto (até 8): `id=<i> color=<c> x=<col> y=<row> size=<n>`.
   (Inclui x,y; remove a exibição enganosa de `size` isolado — mantém `size` mas ao lado de x,y.)
2. **Distâncias par-a-par** — para cada par de objetos, `d(i,j)=|xi-xj|+|yi-yj|` (Manhattan).
   Dá ao LLM o "avatar↔alvo" de graça. Limitar a ~10 pares (objetos menores primeiro) pra não
   estourar o prompt em cenas com muitos objetos.
3. **Resumo de grid** — um mapa coarse célula→cor dominante (ex.: divide o campo em células e
   reporta a cor predominante de cada) pro raciocínio de arranjo em jogos tipo grid. Derivado das
   posições dos objetos (não do frame bruto — mantém o contrato do `state`).
4. **Guia + few-shot** — dizer que a meta costuma ser espacial (aproximar / alinhar / casar um
   arranjo) e dar 2 exemplos de `reward_function` que usam **posição/distância**, não contagem:
   - distância: `reward = -min_manhattan(avatar, targets)` (menor distância = mais perto).
   - arranjo: reward baseada em quantas células/objetos batem uma configuração espacial relativa.
5. **Manter as REGRAS existentes** (reward graduado, sem magic-number, goal_flag raro) — elas
   continuam válidas e o `accept_reward` continua filtrando.

### Contrato do `state` (inalterado)

O `state` continua `list[(shape_hash, {"x","y","color","shape"})]`. Rewards existentes
(iteram objetos) seguem funcionando. A mudança é só no que o PROMPT mostra — a reward passa a
**poder** (e ser induzida a) referenciar x,y e distâncias que já estão no `state`.

## Arquitetura / isolamento

- Mudança contida a **`_build_reward_prompt`** (uma função) + helpers puros (cálculo de
  distâncias par-a-par, resumo de grid) que podem morar no mesmo arquivo ou em `dsl.py` se já
  houver primitiva de distância (`manhattan` existe em `dsl.py`).
- Sem mudança no `state`, no `compile_reward`, no `accept_reward`, nem na pilha de decisão.
- Default-safe: roda no mesmo caminho que já roda sob `CAUSAL_LLM` — sem novo toggle.

## Testes (offline, FakeLLM)

1. `_build_reward_prompt(scene)` **contém x,y** de cada objeto (não só cor/size).
2. O prompt **contém pelo menos uma distância par-a-par** quando há ≥2 objetos.
3. O prompt **contém o resumo de grid** e os few-shot espaciais (distância + arranjo).
4. Uma reward espacial de exemplo (`reward = -manhattan(a,b)`) **passa** `static_reward_check` +
   `accept_reward` sobre estados sintéticos com objetos em posições variadas (gradiente real).
5. Regressão: uma reward de contagem-de-cor constante continua **rejeitada** por `accept_reward`
   (não quebramos o hardening).

## Fora de escopo (YAGNI / follow-up)

- Expor o **frame 64×64 bruto** no `state` pra casamento de padrão pixel-exato (só se o resumo
  de grid derivado de objetos se mostrar insuficiente num jogo de grid real).
- Mudar a percepção pra segmentar células de tabuleiro que hoje mergeiam num só componente.

## Limitação honesta

Recordings são do nosso agente (0 níveis) → nunca vemos estado resolvido, então a reward
espacial ainda **infere** o alvo (não o confirma). O lever torna a meta *expressável* e
*perseguível por gradiente* (distância diminui = progresso), o que a contagem-de-cor nunca
permitiu — mas cruzar nível de fato só se confirma no Kaggle.

## Validação

Offline não confirma (é o wiring/prompt); a validação real é o run offline no Kaggle (RTX Pro
6000, 32B) observando `reward_src` virar espacial (usa x,y/distância), `iw_goal_hits > 0`,
`rprog_fires` com gradiente, e — o teste decisivo — `levels_completed > 0`.
