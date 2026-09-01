import numpy as np

from agents.causal.perception import parse
from agents.causal.policy import candidates, click_key


def _scene_with_bar_and_button():
    # grid 64x64: barra HUD cor 7 na linha 0 (larga) + botao cor 9 pequeno (3x3)
    g = np.full((64, 64), 3, dtype=int)   # fundo cor 3
    g[0:12, :] = 7                         # barra topo cor 7 (bloco enorme)
    g[24:44, 24:44] = 9                    # botao cor 9 (bloco medio)
    return parse(g.tolist())


def test_click_key_distinguishes_bar_from_button():
    scene = _scene_with_bar_and_button()
    kbar = click_key(scene, 30, 0)     # clique na barra (x=30,y=0) -> cor 7, bloco grande
    kbtn = click_key(scene, 31, 31)    # clique no botao (x=31,y=31) -> cor 9, bloco pequeno
    assert kbar != kbtn                # chaves DIFERENTES -> aprende produtividade separada
    assert "9" in kbtn                 # a chave do botao referencia a cor 9
    assert "7" in kbar                 # a chave da barra referencia a cor 7


def test_click_key_same_object_same_key():
    scene = _scene_with_bar_and_button()
    # dois cliques na MESMA barra -> MESMA chave (nao 77 chaves unicas como o v1)
    assert click_key(scene, 5, 0) == click_key(scene, 50, 0)


def test_candidates_clickmap_uses_color_size_key():
    scene = _scene_with_bar_and_button()
    from arcengine import GameAction
    cands = candidates(scene, [GameAction.ACTION6], clickmap=True)
    keys = {c.key for c in cands}
    # com clickmap, as chaves de clique sao por cor/tamanho (poucas), nao 36 celulas
    assert any("7" in k for k in keys) and any("9" in k for k in keys)
    assert len(keys) < 36              # colapsa as 36 celulas em poucas classes
