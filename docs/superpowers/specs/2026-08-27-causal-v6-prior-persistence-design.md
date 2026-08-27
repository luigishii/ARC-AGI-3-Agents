# CausalObjectAgent v6 — Persistência do TransferPrior em disco · Design

> Passo 4b da Fase 5 (fecha o reuso inter-jogos). O v5 entregou o
> `TransferPrior` abstrato (features generalizáveis → produtividade,
> thread-safe, singleton `shared_prior()`), já serializável (`to_dict`/
> `from_dict`). Falta persistir/carregar em disco para (a) um jogo NÃO VISTO
> começar já sabendo quais modalidades são produtivas (warm-start) e (b)
> shipar um prior PRÉ-TREINADO read-only no notebook Kaggle.

## Decisões aprovadas

- **Escrita gated:** carrega o prior no init **sempre**; só **salva** se a env
  `CAUSAL_PRIOR_SAVE` estiver setada. Eval Kaggle = read-only (sem escrita, sem
  concorrência de disco, sem internet); treino local = com a flag acumula.
- **Só o mecanismo** neste ciclo: load/save/merge + wiring, testados. **Não**
  gerar/commitar um `prior.json` real agora (pré-treino fica para depois, p.ex.
  na Fase 6). Sem o arquivo, o agente funciona igual (load é no-op).

## Objetivo

Dar ao `TransferPrior` persistência em disco (JSON) com escrita atômica e
carga única no singleton compartilhado, seguindo a convenção de env do
`CAUSAL_LOG`. Sem LLM/GPU, numpy/stdlib puro, Kaggle-submittable.

## Arquitetura

Mudanças em `agents/causal/transfer.py` (save/load/merge + carga única) e
`agents/causal/agent.py` (load no init, save no cleanup). `policy.py`,
`perception.py`, `hud.py`, `causal_model.py`, `novelty.py`,
`instrumentation.py` **não mudam**.

### 1. `transfer.py` — merge, save, load, carga única

```python
import json
import os
import threading

DEFAULT_PRIOR_PATH = "agents/causal/prior.json"


class TransferPrior:
    # ... (existente: observe, productivity, to_dict, from_dict, _lock) ...

    def merge(self, other: "TransferPrior") -> None:
        with self._lock:
            for feat, (np_, nt) in other.to_dict()["counts"].items():
                c = self._counts.setdefault(feat, [0, 0])
                c[0] += np_
                c[1] += nt


def save_prior(prior: TransferPrior, path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prior.to_dict(), f)
    os.replace(tmp, path)          # atômico


def load_prior(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return TransferPrior.from_dict(json.load(f))


_load_lock = threading.Lock()
_loaded = False


def load_shared_once(path: str) -> None:
    global _loaded
    with _load_lock:
        if _loaded:
            return
        _loaded = True
        disk = load_prior(path)
        if disk is not None:
            shared_prior().merge(disk)
```

- **`merge`** acumula contagens de `other` no self (sob lock). Usa
  `other.to_dict()` para ler com segurança de thread.
- **`save_prior`** escreve atômico (temp + `os.replace`); cria o diretório se
  preciso. Grava o singleton **como está** — que, por já ter sido carregado do
  disco no init, contém `disco + run`; NÃO refunde o disco no save (evita
  dupla-contagem). Acúmulo entre runs vem do **load no início**, não do save.
- **`load_shared_once`** funde o arquivo no singleton **uma vez** por processo
  (guardado por `_load_lock` + `_loaded`), seguro para as threads paralelas do
  `Swarm`. Arquivo ausente → no-op.
- **`reset_shared_prior()`** (já existente) passa a zerar também `_loaded`
  (isolamento de teste — senão o flag global vaza entre testes).

### 2. Wiring no `agent.py`

- `_init_causal_state`: após `self._prior = shared_prior()`, chamar
  `load_shared_once(os.environ.get("CAUSAL_PRIOR", DEFAULT_PRIOR_PATH))`. Como
  `load_shared_once` funde no MESMO singleton que `self._prior` referencia, o
  agente passa a ver o prior pré-treinado.
- Override de `cleanup`:

```python
    def cleanup(self, scorecard=None):
        if os.environ.get("CAUSAL_PRIOR_SAVE"):
            from .transfer import save_prior, DEFAULT_PRIOR_PATH
            save_prior(self._prior, os.environ.get("CAUSAL_PRIOR", DEFAULT_PRIOR_PATH))
        super().cleanup(scorecard)
```

Salva só sob a flag; senão o cleanup é o do base, inalterado.

## Fluxo de dados

Início do processo → 1º agente chama `load_shared_once(path)` → funde o
`prior.json` (se existir) no singleton → todos os agentes/threads começam com o
prior pré-treinado → jogam, alimentando o singleton (v5) → no cleanup, se
`CAUSAL_PRIOR_SAVE`, cada agente grava o singleton atômico no disco (acúmulo
entre runs vem do load).

## Erros e casos de borda

- **Arquivo ausente/inválido:** `load_prior` retorna `None` (checa
  `os.path.exists`); `load_shared_once` no-op. Agente funciona sem prior.
- **Escrita atômica:** `os.replace` é atômico no mesmo filesystem; um leitor
  nunca vê arquivo parcial.
- **Saves concorrentes (threads do Swarm):** todos gravam o MESMO singleton
  crescente; `os.replace` garante que cada arquivo final é íntegro; o último a
  gravar tem o singleton mais completo (compartilhado). Aceitável.
- **Dupla-contagem:** evitada porque o save NÃO refunde o disco (o singleton já
  o contém desde o load).
- **Kaggle read-only:** sem `CAUSAL_PRIOR_SAVE` → nunca escreve; só lê o
  `prior.json` shipado. Sem internet, sem concorrência de disco.
- **Isolamento de teste:** `reset_shared_prior()` zera singleton **e** `_loaded`.

## Testes (TDD, `tests/causal/`)

1. **`save_prior`/`load_prior` roundtrip** (`test_transfer_persistence.py`,
   `tmp_path`): salva um prior com contagens, recarrega, `to_dict` bate;
   arquivo existe.
2. **escrita atômica:** após `save_prior`, não sobra `path+".tmp"`; conteúdo é
   JSON válido.
3. **`load_prior` de caminho inexistente → `None`** (sem exceção).
4. **`merge` acumula:** dois priors com contagens → soma por feature.
5. **`load_shared_once` uma vez:** com um arquivo semente, funde no singleton
   (productivity reflete o disco); 2ª chamada não duplica; `reset_shared_prior`
   permite recarregar.
6. **integração no agente** (`test_agent_persistence.py`): com `CAUSAL_PRIOR`
   apontando p/ um arquivo semente, o agente recém-init vê o prior carregado
   (`productivity` do disco); com `CAUSAL_PRIOR_SAVE` setado, `cleanup()` grava
   o arquivo; sem a flag, `cleanup()` não grava.
7. **Regressão:** os 96 testes v1–v5 seguem verdes.

## Fora de escopo

- Gerar/commitar um `prior.json` pré-treinado (treino offline — depois).
- Merge-com-disco no save / lock de arquivo cross-processo (só relevante p/
  runs paralelos em processos distintos, fora da eval Kaggle).
- Fase 6 (notebook de submissão Kaggle) — consumirá este mecanismo.

## Critério de pronto

- `save_prior`/`load_prior`/`merge`/`load_shared_once` corretos e thread-safe;
  agente carrega no init e salva no cleanup só sob flag; sem arquivo, funciona
  igual ao v5.
- 96 testes v1–v5 + novos verdes.
- Demonstração: rodar 2 processos em sequência com `CAUSAL_PRIOR_SAVE=1`
  apontando o mesmo arquivo e ver as contagens **acumularem** entre eles
  (evidência de persistência cross-run).
