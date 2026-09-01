# ARC-AGI-3 — Enciclopédia dos 25 Jogos Públicos (decodificados do fonte)

> Decodificado lendo `environment_files/<id>/<id>.py` (fonte ofuscado, mesma engine `arcengine`).
> **Uso:** entender a DISTRIBUIÇÃO de mecânicas → prior/few-shot generalizável no modelo.
> A eval é NÃO-vista; não é decorar, é aprender as CLASSES e o que o agente precisa saber.

---

## 🧭 TAXONOMIA (6 classes cobrem os 25) + padrões universais

**Padrões que valem para TODOS os 25 (o mais importante):**
1. **Barra de HUD = orçamento de movimentos** (linha 0, linha 63, ou coluna x=62/63). Decrementa por ação → `lose()` no 0. **NUNCA é a meta. Clicar nela = desperdício.** Cores variam (5,7,14,4,11,6...).
2. **Vitória é sempre GROUNDED espacial/atributo** — posição exata (`avatar.x==goal.x`), match de padrão célula-a-célula, ou contagem-em-zona. **NUNCA é contagem-de-cor agregada.** → nossa reward sintetizada (chutes de contagem/proximidade de cor) é **estruturalmente incompatível**.
3. **Espaço de ação RICO:** teclado (1-4 move/cicla), **ACTION5 (validar/grab/ciclar)**, ACTION6 (clique-seleciona), **ACTION7 (undo)**. Nosso agente colapsou em "só ACTION6 cego" → errado na maioria.
4. **Interação em 2 fases:** clicar-selecionar → mover/posicionar. Não é clique único.
5. **Só sprites específicos são interativos** (tag `sys_click` / cores específicas). Clicar fundo/HUD/parede = ação perdida → **lever clickmap é o fix #1**.
6. **Baselines L0 são PEQUENOS (2–13 ações)** — são puzzles solúveis; nossos 200 desperdiçados = falha de exploração, não de dificuldade.

**As 6 classes:**
| Classe | Jogos | Vitória |
|---|---|---|
| **A. Navegação avatar→alvo** (teclado) | dc22, g50t, tu93, sc25, ls20, m0r0 | avatar na posição-alvo (às vezes casando forma/cor/rotação) |
| **B. Sokoban / empurrar-blocos** | ka59, wa30, su15 | todas as caixas nas células-alvo |
| **C. Manipulação/posicionamento de peças** (selec+mover+girar) | vc33, ar25, cn04, r11l, s5i5, lf52, lp85 | peças cobrem/conectam/alinham aos alvos |
| **D. Preenchimento/pintura por padrão** | cd82, ft09, re86 | grid pintado == padrão-alvo célula-a-célula |
| **E. Sequência/ordenação + VALIDAR** | sb26, tr87, sk48, tn36 | sequência montada casa o alvo; exige ação de RUN/validar |
| **F. Roteamento de fluxo** | sp80 | todos os baldes cheios, sem transbordo |

---

## Os 25 jogos

### A — Navegação avatar→alvo
- **dc22** (bridge-builder): ações [1,2,3,4,6]. Avatar move c/ 1-4; botões de clique (cor 9/10/8) constroem pontes. **Vitória:** avatar(tag `jfva`)==alvo(`goknoi`) posição exata. Budget linha 63.
- **g50t** (ghost-echo): ações [1,2,3,4,5]; 5=UNDO. Avatar cor 9; ecos repetem sua trilha (perigo). Timer = sprite rolante (não barra). **Vitória:** avatar adjacente à flag cor 9.
- **tu93** (routing): ações [1,2,3,4] = direção; vira só rumo a corredor (pixel 2). Avatar cor 9. **Vitória:** todos avatares nas saídas cor 14. Budget linha 63 cor 6.
- **sc25** (mago-rúnico): ações [1,2,3,4,6]. Avatar cor 10/9 move; grade de runas cor 12 (clique) casa padrão 3×3 → auto-lança feitiço que **remove obstáculo**. **Vitória (real):** avatar encosta na saída `exydhv` (cor 9+10). NÃO é lights-out. Budget coluna x=62 cor 14.
- **ls20** (shape-shifter maze): ações [1,2,3,4] teclado, 5px/tecla. Avatar bicolor 12/9; botões de piso mudam rotação/cor/forma. **Vitória:** avatar em cada alvo cor 5 **casando forma+cor+rotação**. 3 vidas cor 8; budget y=61.
- **m0r0** (movimento espelhado): ações [1,2,3,4,5,6]. Setas movem vários marcadores cor 10 em direções ESPELHADAS; clique pega caixa cor 9. **Vitória:** pares de marcadores convergem/pareiam. Traps cor 8. Budget 150.

### B — Sokoban / empurrar-blocos
- **ka59**: ações [1,2,3,4,6]. Clique(6) seleciona bloco (centro vira cor 0); setas empurram. **Vitória:** cada bloco na moldura-alvo compatível (`0010`) + bloco especial no seu alvo. Inimigos `Enemy` matam. Budget linha 63 cor 4; StepCounter 100-200.
- **wa30** (sokoban-ímã): ações [1,2,3,4,5]; avatar cor 14; **5=grab/attach/release**. Caixas recoloridas como DICA (4=livre,5=presa,3=adjacente,**0=resolvida**). **Vitória:** todas as caixas nos alvos cor 2 e soltas. Budget linha 63 cor 7. → contagem caixas-em-alvo = bom reward denso.
- **su15** (onda de choque): ações [6,7]; 7=undo caro. Clique cria onda que empurra blocos-número (fusão 2048: colisão soma) rumo à faixa-topo (linhas 0-9). **Vitória:** contagem exata de blocos por valor/tipo na zona == `data["xkstxyqbs"]`. Empurrar = clicar do lado OPOSTO ao destino.

### C — Manipulação/posicionamento de peças
- **vc33** (gravity align): [6] click-only. Clicável = cor 9 (botão) + cor 1 (bloco). Barra topo cor 7 = budget. **Vitória:** marcadores 11/14/15 alinhados a alvos de mesma cor sob `Gravity`. **Nosso agente clicava a barra cor-7 inerte → 0.** L0=7.
- **ar25** (tangram-cobertura): [1..7]; 1-4 move peça, 5 cicla seleção, 6 clique, **7=UNDO**. Peças giram ao cruzar eixos cor 10. **Vitória:** todos marcadores cor 11 cobertos. Budget coluna x=63.
- **cn04** (jigsaw conectores): 1-4 move, 5 rotaciona, 6 clique-seleciona. **Vitória:** todos os pinos cor 8/13 pareados em células coincidentes. Budget linha 0 centrado.
- **r11l** (block placement): [6] click-only. Clica peça (cor 3) → clica vazio move; o "carregador" segue o centroide. **Vitória:** cada carregador colide seu alvo de mesma cor. Hazards `defgjl` (5×=lose). Budget topo, 60.
- **s5i5** (braço articulado): [6]; clicar extremos das barras estende/encurta, pivôs rotacionam 90°. **Vitória:** cada efetuador(`0064`) na posição EXATA do alvo(`0087`). Budget StepCounter.
- **lf52** (peg-solitaire): [1,2,3,4,6,7]; 1-4 só rolam câmera (ruído!), 6 clique 2-fases (peça→marcador salta 2 capturando), 7 undo. **Vitória:** reduzir pegs cor 14 a 1 (ou 2). Budget linha 0.
- **lp85** (anéis rotativos): [6] click-only. Botões-seta cor 8=CCW / cor 14=CW giram o anel inteiro 1 nó. **Vitória:** tokens girados até casar as âncoras fixas cor 11/12. L0=5 cliques. Budget topo, 13.

### D — Preenchimento/pintura por padrão
- **cd82** (paint-to-match): [1,2,3,4,5,6]; clica cor na paleta → posiciona emissor (anel 8-pos) → **5=disparar** (pinta faixa). **Vitória:** grid pintado == padrão-alvo OCULTO (exclui diagonais). Budget linha 63 (4/5), 100.
- **ft09** (Lights-Out de cor): [6] click-only. Clicar tile propaga cor aos vizinhos em cruz (toggle paleta [9,8]). **Vitória:** cada alvo `bsT` satisfeito pela relação de cor dos 8 vizinhos (padrão 3×3). Budget linha 63 cor 12/11. **Meta é arranjo célula-a-célula, contagem-de-cor não distingue.**
- **re86** (snake tiling): [1,2,3,4,5]; 1-4 move peça ativa (3px), 5 troca ativa. Peça CRESCE ao bater em parede (tipo cobra). **Vitória:** peças preenchem os contornos-alvo. Budget StepCounter 100-400.

### E — Sequência/ordenação + VALIDAR
- **sb26** (token sequence): [5,6,7]; 6 clique (seleciona/troca/coloca token), **5=RUN/validar**, 7=undo grátis. Tokens coloridos base + fileira-alvo no topo (ordem de cor). **Vitória:** rodar (5) e o marcador atravessar a sequência casada. Barra energia y=53, 64.
- **tr87** (gramática substituição): [1,2,3,4] teclado; 1/2 cicla variante da peça, 3/4 move cursor. **Vitória:** fileira-baixo traduzida pelas regras == fileira-cima. Budget linha 63 (1/4), 128/256. Sem clique.
- **sk48** (replicar seq. de cor): [1,2,3,4,6,7]; setas movem tubo/cobra (estende/retrai), 6 seleciona cabeça, 7 undo. **Vitória:** tubo de controle cobre a MESMA sequência de cores-alvo que o de referência. Budget y=53 cor 2/3, 196.
- **tn36** (síntese de programa): [6] click-only. Clicável = segmentos `Maidxz` (cor 1/5, togglam bits do PROGRAMA), botão run `sucqgk`, abas `tozzsf`. Bitmask→instrução (mover/girar/escalar) move o bloco. **Vitória:** bloco casa alvo (pos+escala+rot+cor). **Nosso agente martelava a barra rolante inerte → fixação.** L0 curto.

### F — Roteamento de fluxo
- **sp80** (encanamento): [1,2,3,4,5,6]; 6 seleciona barra(cor 8)/defletor(cor 15), 1-4 move, **5=escoar** (≤4 tentativas). Água nasce cor 4 (topo), cai; barras espalham. **Vitória:** todos os baldes cor 11→13 cheios, sem tocar dreno cor 1. Budget linha 0 cor 14. (Alguns níveis rotacionam a tela 180°.)

---

## 🎯 IMPLICAÇÕES CONCRETAS PRO AGENTE (o que fazer)
1. **clickmap (fix #1):** aprender produtividade do clique por (cor, tamanho do bloco) → **parar de clicar HUD/fundo/parede** e focar no interativo (cor 9/1/etc). Resolve vc33/tn36/r11l/lp85/su15 direto e generaliza.
2. **Usar o espaço de ação inteiro:** detectar jogos de TECLADO (1-4) e usar **ACTION5 (validar/grab/ciclar)** e **ACTION7 (undo)**. Colapsar em ACTION6 mata ~metade dos jogos (todos de classe A/B/E de teclado).
3. **Reward GROUNDED, nunca contagem-de-cor:** distância-ao-alvo (nav), match célula-a-célula (pintura), contagem-em-zona (su15), contagem-de-objetos (peg/sokoban). Injetar o `_pick_target` grounded + o `MovementModel`.
4. **Prior de classe (few-shot):** dar as 6 classes no prompt do direct → o modelo reconhece "isto é navegação/sokoban/sequência" e infere reward/meta certos.
5. **Mascarar a barra de HUD** (linha 0/63, coluna 62/63): garantir que HudMask pega essas bandas — elas são orçamento, não sinal.
6. **Ação de VALIDAR:** classe E (sb26/tn36) só completa após um RUN (ACTION5/botão); o agente precisa aprender a montar-depois-validar, não validar cedo.
</content>
