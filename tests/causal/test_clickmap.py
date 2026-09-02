import numpy as np

from agents.causal.perception import parse
from agents.causal.policy import candidates, click_key


def _scene_with_bar_and_button():
    # grid 64x64: barra HUD cor 7 na linha 0 (larga, >100px) + botao cor 9 pequeno (<100px)
    g = np.full((64, 64), 3, dtype=int)   # fundo cor 3
    g[0:12, :] = 7                         # barra topo cor 7 (768px, HUD inerte)
    g[30:36, 30:36] = 9                    # botao cor 9 (6x6=36px, interativo)
    return parse(g.tolist())


def test_click_key_distinguishes_bar_from_button():
    scene = _scene_with_bar_and_button()
    kbar = click_key(scene, 30, 0)     # clique na barra (x=30,y=0) -> cor 7, bloco grande
    kbtn = click_key(scene, 31, 31)    # clique no botao (x=31,y=31) -> cor 9, bloco pequeno
    assert kbar != kbtn                # chaves DIFERENTES -> aprende produtividade separada
    assert "c9" in kbtn               # a chave do botao referencia a cor 9
    assert "c7" in kbar               # a chave da barra referencia a cor 7


def test_click_key_different_cells_different_keys():
    scene = _scene_with_bar_and_button()
    # cliques em celulas DIFERENTES do mesmo objeto -> chaves DIFERENTES
    # (preserva info espacial pra distinguir botoes em posicoes diferentes)
    k1 = click_key(scene, 5, 0)       # barra, celula (0,0)
    k2 = click_key(scene, 50, 0)      # barra, celula (4,0)
    assert k1 != k2                   # posicoes distintas -> keys distintas
    assert "c7" in k1 and "c7" in k2  # mesma classe visual (cor 7)


def test_candidates_clickmap_filters_large_objects():
    scene = _scene_with_bar_and_button()
    from arcengine import GameAction
    cands = candidates(scene, [GameAction.ACTION6], clickmap=True)
    keys = {c.key for c in cands}
    # com clickmap, objetos grandes (barra HUD) sao FILTRADOS
    assert not any("c7" in k for k in keys)  # barra (cor 7, size>100) nao aparece
    assert any("c9" in k for k in keys)      # botao (cor 9, size<=100) aparece
    # todos os candidatos restantes tem obj_size <= 100
    assert all(c.obj_size <= 100 for c in cands)
