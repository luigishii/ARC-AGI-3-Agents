# ARC-AGI-3 — Enciclopédia de Jogos (taxonomia de mecânicas)

> **Objetivo:** decodificar os 25 jogos PÚBLICOS lendo o fonte (`environment_files/<id>/<id>.py`),
> extrair a MECÂNICA de cada um, e destilar numa **taxonomia de classes de mecânica**.
> Isso vira **prior/few-shot** injetado no modelo → num jogo NÃO-visto ele reconhece a classe
> e infere a meta mais rápido. NÃO é decorar os 25 (eval é não-vista); é aprender a DISTRIBUIÇÃO.
> Ressalva: usar o fonte é pra ENTENDER a mecânica (offline, pesquisa), não exploit na eval.

## Como cada jogo é decodificado (o que extrair)
- **Ação(ões) disponível(is)** (`available_actions=[...]`): teclado (1-5), clique (6), etc.
- **O que é CLICÁVEL/INTERATIVO** de verdade (sprites com tag `sys_click`; resto é inerte).
- **HUD/orçamento** (barras que decrementam por ação — NÃO são a meta).
- **Condição de VITÓRIA** (o método que dispara `next_level()`).
- **Classe de mecânica** (ver taxonomia embaixo).
- **Baseline de ações do L0** (do scorecard) = pista da solução mínima.

---

## ✅ vc33 — Gravity block-alignment puzzle (DECODIFICADO)
- **Ações:** `available_actions=[6]` → **só clique**.
- **CLICÁVEL (só isto):**
  - **cor 9 (vermelho)** = botões (`sys_click`, tag `0022jvmlspyigc`) → disparam a montagem/movimento.
  - **cor 1 (azul)** = blocos (`sys_click`, tag `0004sttgkofqwb`) → posicionáveis.
- **INERTE (clicar = desperdiça movimento):** fundo cor 3, padding cor 4, paredes cor 5,
  e a **barra do topo (linha 0, cor 7) = CONTADOR DE MOVIMENTOS** (`StepCounter`, 50–200).
- **Gravidade:** cada nível tem vetor `Gravity` (ex.: `[2,0]`,`[0,3]`,`[-3,0]`) → blocos caem/assentam.
- **VITÓRIA (`ielczunthe`):** cada marcador colorido (cores 11/14/15) alinhado ao seu alvo
  de **cor correspondente**, respeitando a gravidade. Perde se `StepCounter` zera.
- **Classe:** **posicionamento/alinhamento sob gravidade, com orçamento de movimentos, clique-só-no-interativo.**
- **Baseline L0 = 7 ações** (7 cliques certos resolvem).
- **⚠️ Por que nosso agente deu 0:** clicou a barra cor-7 (topo) e o fundo — **inerte** — os 200 movimentos.
  Nunca clicou os botões cor-9. **Lição geral: aprender quais alvos de clique MUDAM o frame e focar neles
  (= clickmap da Tufa: produtividade por (cor, tamanho do bloco)).**

---

## ⬜ [id] — [nome] (PENDENTE)
<!-- template: copiar por jogo -->
- **Ações:**
- **Clicável/interativo:**
- **HUD/orçamento:**
- **Vitória:**
- **Classe:**
- **Baseline L0:**
- **Notas p/ o agente:**

_(Repetir pros 24 restantes: sk48, tn36, m0r0, bp35, cn04, dc22, tu93, lp85, ka59, wa30,
lf52, r11l, sc25, sp80, ar25, sb26, cd82, re86, s5i5, ls20, ft09, su15, tr87, g50t)_

---

## Taxonomia de mecânicas (preencher conforme decodifica)
- **Alinhamento/posicionamento sob gravidade** — vc33.
- **Lights-out / toggle célula-a-célula** — (candidato: sc25, decodificado antes por replay).
- **Navegação avatar→alvo (teclado)** — (candidato: ls20, ka59 — avatar anda por tecla).
- **Preenchimento/contador** — ?
- **...**

## Como isso entra no agente (o prior)
1. **clickmap** (produtividade por cor/tamanho do clique) → foca cliques no que muda o frame
   (resolve o desperdício em vc33/tn36 sem saber a regra exata).
2. **few-shot de classes** no prompt do direct: "no AGI-3 os jogos são destas classes; a meta
   costuma ser X; identifique a classe e aja". Prior generalizável, não memorização.
3. **reward por-classe:** alinhamento (vc33), padrão-célula (lights-out), distância (navegação)...
