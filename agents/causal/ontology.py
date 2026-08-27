# agents/causal/ontology.py
# Componente A do redesign Fase-2 (OPINE-World): motor de exploração bayesiano,
# NumPy/stdlib puro, ZERO LLM. Effect signatures (Δ) + tabela local de efeitos
# com prior de Dirichlet + erro de ontologia (η). η alto = "onde explorar agora";
# a CORRETUDE de um modelo é decidida por replay exato, NÃO por η.
from __future__ import annotations

import math

# Alfabeto crescente de assinaturas de efeito (deltas exatos são descartados):
#   no_change · x · y · x,y · recolor · pixels · gone · born


def effect_signature(prev_obj, curr_obj) -> str:
    """Assinatura categórica da mudança de UM objeto pareado. Guarda QUAIS
    atributos mudaram, nunca por quanto (x:12→15 e x:1→60 = mesma assinatura 'x')."""
    if prev_obj is None and curr_obj is None:
        return "no_change"
    if prev_obj is None:
        return "born"
    if curr_obj is None:
        return "gone"
    tokens = set()
    dr = round(curr_obj.centroid[0] - prev_obj.centroid[0])
    dc = round(curr_obj.centroid[1] - prev_obj.centroid[1])
    if dc != 0:
        tokens.add("x")
    if dr != 0:
        tokens.add("y")
    if curr_obj.color != prev_obj.color:
        tokens.add("recolor")
    if curr_obj.shape_hash != prev_obj.shape_hash:
        tokens.add("pixels")
    if not tokens:
        return "no_change"
    return ",".join(sorted(tokens))


def normalized_entropy(probs) -> float:
    """Entropia de Shannon normalizada em [0,1] (H/log K, K=|suporte|).
    K<=1 → 0.0 (sem incerteza possível)."""
    k = len(probs)
    if k <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return h / math.log(k)


class LocalEffectTable:
    """Cada transição de objeto arquivada numa linha j=(τ, a, u) [tipo, ação,
    contexto local]; conta assinaturas por linha; posterior de Dirichlet simétrico
    (α) sobre a distribuição de efeitos da linha. Linha que mistura assinaturas =
    sub-observada ou faltando feature de contexto."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.rows = {}            # j(tuple) -> {sig: count}
        self._alphabet = set()

    def observe(self, tau, a, u, sig) -> None:
        j = (tau, a, u)
        row = self.rows.setdefault(j, {})
        row[sig] = row.get(sig, 0) + 1
        self._alphabet.add(sig)

    def alphabet(self) -> list:
        return sorted(self._alphabet)

    def posterior(self, j) -> dict:
        """P(assinatura | linha) sob Dirichlet simétrico, sobre o alfabeto global."""
        row = self.rows.get(j, {})
        alph = self.alphabet()
        k = len(alph)
        denom = sum(row.values()) + self.alpha * k
        if denom == 0:
            return {}
        return {sig: (row.get(sig, 0) + self.alpha) / denom for sig in alph}

    def row_entropy(self, j) -> float:
        """Incerteza de efeito da linha (entropia normalizada da posterior)."""
        post = self.posterior(j)
        if not post:
            return 0.0
        return normalized_entropy(list(post.values()))

    # alias semântico usado pelo erro de ontologia
    def effect_uncertainty(self, j) -> float:
        return self.row_entropy(j)

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "rows": {"|".join(map(str, k)): dict(v) for k, v in self.rows.items()},
            "alphabet": sorted(self._alphabet),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LocalEffectTable":
        t = cls(alpha=d.get("alpha", 1.0))
        t.rows = {tuple(k.split("|")): dict(v) for k, v in d.get("rows", {}).items()}
        t._alphabet = set(d.get("alphabet", []))
        return t


def ontology_error(type_entropy: float, effect_entropy: float) -> float:
    """Erro de ontologia por objeto = noisy-OR de duas entropias normalizadas:
    incerteza de TIPO do objeto ⊕ incerteza de EFEITO da sua linha.
    η alto marca objetos/contextos que os tipos atuais ainda não explicam."""
    return 1.0 - (1.0 - type_entropy) * (1.0 - effect_entropy)
