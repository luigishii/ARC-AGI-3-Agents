# Guarda Global Anti-Fixação — Design Spec

> **Data:** 2026-08-31 · **Status:** aprovado · **Escopo:** guarda pós-pilha que quebra runs da mesma ação. Default-safe.

## Problema (run offline dos 4 jogos)

A camada de cobertura (`_cover_decide`, lever anterior) foi colocada **no fim da pilha de decisão**
(`LLM → navigate → rprog → IW → plan → η → COVER → greedy`). Resultado ao vivo: **não quebrou a
fixação** — `ls20` ficou em **ACTION4 ×167 seguidas** com `CAUSAL_COVER` ligado. Motivo: as camadas
**acima** (rprog/η/IW/goal) fixam e devolvem sempre a mesma ação; a cobertura só dispara quando
**todas** retornam None → **nunca chega a rodar**. A anti-fixação está no lugar errado.

## Objetivo

Um **guarda GLOBAL** que roda **depois** da pilha inteira decidir: se a **mesma key** foi escolhida
**K vezes seguidas** (fixação), **sobrepõe** a decisão com uma escolha de cobertura **excluindo a key
fixada**. Pega fixação **venha de qual camada for**. Reusa `_cover_decide` (menos-visitada) pra a
substituta variar em vez de qualquer uma.

## Escopo

**Dentro:** estado `_fix_run`/`_fix_breaks`/`_fix_on`/`_fix_k` (`_init`); guarda em `choose_action`
(entre o RESET-check e `action = cand.action`); 1 chave em `phase2_stats` (`fix_breaks`); toggle
`CAUSAL_FIX` nos 2 builders. Só `agents/causal/agent.py` + os 2 builders + testes.

**Fora:** atacar a causa upstream (recência no rprog/η — follow-up se o guarda não bastar);
mudar as camadas existentes.

## Componentes

### 1. Estado (`_init_causal_state`)

```python
self._fix_run = 0             # repetições consecutivas da MESMA key escolhida
self._fix_breaks = 0          # diag: vezes que o guarda quebrou uma fixação
self._fix_on = os.environ.get("CAUSAL_FIX", "0") != "0"
self._fix_k = int(os.environ.get("CAUSAL_FIX_K", "3"))
```

### 2. Guarda em `choose_action`

Inserido **entre** o `if cand is None: return RESET` (cand garantido não-None) e `action = cand.action`
(`agent.py:273-274`). Compara com `self._last_key` — que nesse ponto ainda é a key da **jogada
anterior** (só atualiza em `agent.py:301`).

```python
        if cand.key == self._last_key:
            self._fix_run += 1
        else:
            self._fix_run = 0
        if self._fix_on and self._fix_run >= self._fix_k and cands:
            alt = [c for c in cands if c.key != cand.key]
            if alt:
                cand = keymap.get(self._cover_decide(alt), cand)
                self._fix_breaks += 1
                self._fix_run = 0
```

Efeito com `FIX_K=3`: padrão vira `[fix, fix, fix, VARIADA, …]` — injeta ~1 ação de cobertura a cada
3 repetições, quebrando a fixação de 100% e alimentando o sweep. A substituta vem de `_cover_decide`
(menos-visitada, excluindo a key fixada).

### 3. Diagnóstico (`phase2_stats`)

```python
"fix_breaks": self._fix_breaks,
```

Alto = o guarda está ativamente quebrando fixações.

### 4. Toggle no notebook

`build_notebook.py` e `build_offline_notebook.py`: adicionar `"CAUSAL_FIX=1\n"` ao `.env` junto de
`CAUSAL_COVER`.

## Comportamento e segurança

- **Default-safe:** guarda só sob `CAUSAL_FIX` (default off) → decisão idêntica sem o toggle.
- **`_last_key` intacto:** o guarda **lê** `_last_key` (jogada anterior) mas não o altera; a atualização
  em `agent.py:301` passa a refletir a key possivelmente **sobreposta** (correto — é a ação real tomada).
- **Sem alt disponível:** se todas as candidatas forem a key fixada (ex: 1 só candidata), o guarda não
  força (mantém cand) — não trava.
- **Reusa `_cover_decide`** (lever anterior) → a substituta varre o espaço.
- **Sem GPU/LLM aqui:** testável com Candidates sintéticos + `_last_key` setado à mão.

## Testes (TDD)

1. quebra após K: com `_fix_on`, `FIX_K=3`, `_last_key="A"`, chamar o guarda com `cand.key="A"` 3× →
   na 3ª repetição sobrepõe por uma candidata **diferente** de "A"; `fix_breaks` incrementa.
2. abaixo de K não quebra: 2 repetições < K → mantém `cand` (key "A"), `fix_breaks` inalterado.
3. key diferente zera o run: alternando "A","B" nunca acumula → nunca quebra.
4. sem alternativa: única candidata "A" repetida → não força (mantém "A").
5. off por default: sem `CAUSAL_FIX` → nunca sobrepõe, mesmo com run alto.
6. `phase2_stats` expõe `fix_breaks`.
7. builders: o `.env` gerado contém `CAUSAL_FIX=1`.

## Entregável

`agent.py` + os 2 builders + testes verdes (baseline atual + N). Notebook offline reembala `agent.py`
sozinho. **Validação real (offline multi-jogo):** observar `fix_breaks > 0` (quebrando fixação) e se
`ls20` para de martelar ACTION4 e se `levels_completed` sobe. Se quebrar mas não cruzar → a causa
upstream (rprog/η propondo sempre a mesma) é forte → próximo lever = recência no rprog/η.
