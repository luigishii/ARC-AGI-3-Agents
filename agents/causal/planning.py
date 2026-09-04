# agents/causal/planning.py
from __future__ import annotations

PLAN_DEPTH = 3
PLAN_BEAM = 8


class TransitionModel:
    def __init__(self):
        self.trans = {}    # sig -> {key -> {next_sig -> count}}

    def observe(self, prev_sig, key, next_sig) -> None:
        d = self.trans.setdefault(prev_sig, {}).setdefault(key, {})
        d[next_sig] = d.get(next_sig, 0) + 1

    def predict_next(self, sig, key):
        d = self.trans.get(sig, {}).get(key)
        if not d:
            return None
        return max(d.items(), key=lambda kv: kv[1])[0]

    def known_keys(self, sig):
        return list(self.trans.get(sig, {}).keys())

    def to_dict(self) -> dict:
        return {s: {k: dict(nn) for k, nn in kk.items()}
                for s, kk in self.trans.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionModel":
        m = cls()
        m.trans = {s: {k: dict(nn) for k, nn in kk.items()}
                   for s, kk in d.items()}
        return m


def _sig_dist(a, b):
    sa = set(a.split(";")) if a else set()
    sb = set(b.split(";")) if b else set()
    return len(sa ^ sb)


def _terminal_score(sig, frontier, novelty, anchors):
    if anchors:
        if frontier:
            return -0.5                     # desconhecido: entre "na âncora" (0) e "longe"
        return -min(_sig_dist(sig, a) for a in anchors)
    if frontier:
        return 1.0
    return novelty.novelty(sig)


def plan(start_sig, start_keys, tmodel, novelty, anchors,
         depth=PLAN_DEPTH, beam=PLAN_BEAM, key_prior=None):
    """key_prior(key)->[0,1]: produtividade aprendida da chave-raiz. So pesa na FRONTEIRA
    (transicao desconhecida): sem isso toda fronteira empata em 1.0 e o max() devolve a
    1a da lista (= raster), gastando acoes em chaves ja vistas como inertes."""
    if not start_keys:
        return None
    # exige ao menos uma transição conhecida a partir do estado atual
    if all(tmodel.predict_next(start_sig, k) is None for k in start_keys):
        return None

    # nó = (first_key, sig_atual, frontier?)
    nodes = []
    for k in start_keys:
        nxt = tmodel.predict_next(start_sig, k)
        nodes.append((k, nxt if nxt is not None else start_sig, nxt is None))

    def score(node):
        first, sig, frontier = node
        base = _terminal_score(sig, frontier, novelty, anchors)
        if frontier and key_prior is not None:
            base += 0.5 * (float(key_prior(first) or 0.0) - 0.5)   # [-0.25, +0.25]
        return base

    for _ in range(1, depth):
        nodes.sort(key=score, reverse=True)
        nodes = nodes[:beam]
        nxt_nodes = []
        for (first, sig, frontier) in nodes:
            keys = [] if frontier else tmodel.known_keys(sig)
            if not keys:
                nxt_nodes.append((first, sig, frontier))    # terminal (fronteira ou beco)
                continue
            for k in keys:
                nn = tmodel.predict_next(sig, k)
                nxt_nodes.append((first, nn if nn is not None else sig, nn is None))
        nodes = nxt_nodes

    return max(nodes, key=score)[0]
