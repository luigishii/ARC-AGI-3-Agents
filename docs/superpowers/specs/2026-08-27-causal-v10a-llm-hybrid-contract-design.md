# CausalObjectAgent v10a — LLM Hybrid: contrato + meta declarativa · Design

> Passo 7 (pivot pra opção D). v1–v9 (determinístico numpy) dá 0 níveis — falta
> **inferência semântica da meta**. Objetivo agora: **score/premiação**. Um LLM
> aberto offline (Qwen2.5-Coder-7B-Instruct) infere a meta (macro) e nossa
> fundação causal executa/valida em numpy (micro).

## Restrição honesta que molda o escopo

**Este ambiente não tem GPU/LLM/internet.** O TDD local cobre apenas o
**encanamento determinístico** (montar prompt, parsear saída, executar meta,
fallback) usando um **LLM mockado** (`FakeLLM`). O modelo real (serving vLLM/
transformers, pesos como Kaggle Dataset) e a validação de raciocínio ficam para
**v10b** (no container Kaggle). O v10a não instancia nenhum modelo.

## Decisões aprovadas

- **Saída do LLM = meta declarativa (JSON)** (código/rule em sandbox fica p/ v10c).
- **Modelo alvo = Qwen2.5-Coder-7B-Instruct** (definido p/ o v10b).
- **Serving abstraído** — v10a só define a interface `LLMClient`.

## Escopo do v10a (este spec)

Só o **contrato LLM**, testável localmente com mock:
1. `LLMClient` (interface) + `NullLLMClient` (no-op → hybrid inerte, cai no v9/v8).
2. `build_prompt(scene, dynamics)` — serializa nossa percepção objeto-cêntrica +
   dinâmica aprendida num prompt pedindo a meta.
3. `parse_goal(text)` — extrai/valida o JSON de meta da resposta do LLM.
4. `execute_goal(goal, scene, moves)` — traduz a meta numa `action_key` concreta.

**Fora do v10a:** serving real + notebook + pesos (v10b); sandbox de código
(v10c); o controlador de quando-chamar/validar/re-perguntar e o wiring no agente
(v10d) — este spec entrega as peças puras que o v10d vai orquestrar.

## Arquitetura

Novo módulo `agents/causal/llm.py`. Nenhum módulo existente muda (v10a é
aditivo e não é ligado ao agente ainda — o wiring é o v10d). Reusa
`state_signature`/`cell_center` conforme necessário.

### 1. `LLMClient` (interface abstrata)

```python
class LLMClient:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError

class NullLLMClient(LLMClient):
    def complete(self, prompt: str) -> str:
        return ""            # sem modelo → sem meta → fallback determinístico
```

Testes usam `FakeLLM(canned: str)` → devolve `canned`. A impl real
(`VLLMClient`/`HFClient`) é v10b, instanciada só no Kaggle.

### 2. `build_prompt(scene, dynamics) -> str`

Serializa **estado estruturado** (não pixels):

- dimensões da grade; nº de objetos;
- por objeto: `id, cor, bbox, centroide, tamanho` (HUD já mascarado pela
  percepção);
- `available_actions`;
- **dinâmica aprendida** (`dynamics`, um dict que o v10d montará do
  `CausalModel`/`MovementModel`): ex. `moves` (ação→(dr,dc)), efeitos de clique
  notáveis, chaves de progresso.

Instrui o LLM a responder **só** um JSON de meta (ver schema). Determinístico
(mesma entrada → mesmo prompt) → testável por substring.

### 3. Schema de meta + `parse_goal(text) -> dict | None`

Extrai o **primeiro bloco `{...}`** da resposta, `json.loads`, valida `type`.
Tipos suportados no v10a:

- `{"type":"press","action":"ACTION1"}` — apertar uma ação simples.
- `{"type":"click_cell","gx":G,"gy":G}` — clicar numa célula da grade 6×6.
- `{"type":"reach","avatar":<sel>,"target":<sel>}` — mover o avatar até o alvo.
  `<sel>` (seletor de objeto) = `{"id":I}` | `{"color":C}` | `"rarest"`.

`parse_goal` retorna `None` se: não há JSON, JSON inválido, `type` desconhecido,
ou campos faltando (robusto a alucinação → cai no fallback).

### 4. `execute_goal(goal, scene, moves) -> action_key | None`

Traduz a meta numa `action_key` concreta (que o agente mapeia p/ `Candidate`):

- `press` → `goal["action"]` se está em `available` (o v10d filtra), senão `None`.
- `click_cell` → `f"ACTION6@cell={gx},{gy}"`.
- `reach` → resolve `avatar` e `target` na cena pelos seletores; escolhe a ação
  de `moves` (dict `action→(dr,dc)`) que mais reduz a distância Manhattan
  avatar→alvo; `None` se não há avatar/alvo/movimento útil. (Mesma lógica gulosa
  do `navigate` do v9, mas com avatar/alvo **ditados pelo LLM**.)

`moves` é injetado (o v10d o obterá do `MovementModel`/`CausalModel`), mantendo
`execute_goal` puro e testável.

## Fluxo de dados (visão futura, v10d)

`build_prompt(scene, dynamics)` → `LLMClient.complete` → `parse_goal` →
`execute_goal(goal, scene, moves)` → `action_key` → `Candidate` → ambiente. O
micro-loop valida (nível subiu? efeito previsto?); se a meta falha após N ações,
o v10d re-consulta o LLM com a nova evidência. Chamadas ao LLM são **esparsas**
(1 por nível / ao ficar preso), nunca por ação.

## Erros e casos de borda

- **Sem modelo (`NullLLMClient`) / resposta vazia / JSON inválido:** `parse_goal`
  → `None` → sem meta → fallback determinístico (v8/v9). Seguro.
- **Seletor não encontra objeto na cena:** `execute_goal` → `None` → fallback.
- **`reach` sem `moves`:** `None` → fallback.
- **Determinismo:** dado (scene, dynamics) o prompt é fixo; dado (goal, scene,
  moves) o `action_key` é fixo.
- **Robustez a alucinação:** todo parsing valida schema; qualquer desvio → `None`.

## Testes (TDD, `tests/causal/test_llm.py`)

1. **`NullLLMClient.complete` → `""`**; `FakeLLM("...").complete` → o texto.
2. **`build_prompt`** contém as dimensões, os objetos (cor/centroide), as ações
   disponíveis, a dinâmica (`moves`), e a instrução de responder JSON de meta;
   determinístico.
3. **`parse_goal`** aceita cada tipo (`press`/`click_cell`/`reach`) com JSON
   embutido em texto (ex. cercado de prosa); rejeita → `None` p/ JSON ausente,
   inválido, `type` desconhecido, campos faltando.
4. **`execute_goal` press** → a `action_key` da ação; **click_cell** →
   `ACTION6@cell=gx,gy`; **reach** (avatar id/cor e target por seletor) → a ação
   de `moves` que aproxima; casos → `None` (sem avatar/alvo/moves).
5. **seletor `rarest`** resolve o objeto de cor mais rara.
6. **Regressão:** os 128 testes v1–v8 seguem verdes (v10a é aditivo).

## Fora de escopo (próximos sub-projetos)

- **v10b:** serving real (Qwen2.5-Coder-7B via vLLM/transformers), pesos como
  Kaggle Dataset, notebook atualizado, medir latência/caber em 9h.
- **v10c:** saída de **código** (`decide(scene)->action`) em sandbox restrito.
- **v10d:** controlador (quando chamar, validar hipótese, re-perguntar) + wiring
  no `agent.py` (camada LLM acima de navigate→plan→greedy) + `moves` do
  `MovementModel` (do v9).

## Critério de pronto (v10a)

- `agents/causal/llm.py` com `LLMClient`/`NullLLMClient`, `build_prompt`,
  `parse_goal`, `execute_goal`, testados com `FakeLLM`.
- 128 testes v1–v8 + novos verdes.
- **Nada de modelo real** — validação de raciocínio é v10b (Kaggle).
