from types import SimpleNamespace

from agents.causal.llm import build_class_prompt, parse_class


def _obj(id=0, color=3, size=4):
    return SimpleNamespace(id=id, color=color, centroid=(1, 1), size=size, bbox=(1, 1, 2, 2))


def _scene(*objs):
    g = [[0] * 64 for _ in range(64)]
    return SimpleNamespace(objects=list(objs), grid=g)


def test_class_prompt_has_taxonomy_and_grid():
    p = build_class_prompt(_scene(_obj()), {"available": ["ACTION1", "ACTION6"]})
    for c in "ABCDEF":
        assert f"{c}." in p or f"{c} " in p or f'"{c}"' in p
    assert "sokoban" in p.lower()
    assert "GRID" in p
    assert "AVAILABLE_ACTIONS" in p and "ACTION6" in p
    assert '"cls"' in p and '"click"' in p and '"hud_rows"' in p


def test_parse_class_full():
    g = parse_class('{"cls":"A","avatar":9,"target":5,"click":[9,1],'
                    '"hud_rows":[63],"hud_cols":[]}')
    assert g == {"cls": "A", "avatar": 9, "target": 5, "click": {9, 1},
                 "hud_rows": [63], "hud_cols": []}


def test_parse_class_tolerates_fences_and_nulls():
    g = parse_class('```json\n{"cls":"C","avatar":null,"target":null,"click":null}\n```')
    assert g["cls"] == "C"
    assert g["avatar"] is None and g["target"] is None and g["click"] is None
    assert g["hud_rows"] == [] and g["hud_cols"] == []


def test_parse_class_rejects_bad_class_or_garbage():
    assert parse_class('{"cls":"Z"}') is None
    assert parse_class("no json here") is None
    assert parse_class("") is None


def test_parse_class_drops_out_of_range_values():
    g = parse_class('{"cls":"B","avatar":99,"target":-1,"click":[3,"x",200],'
                    '"hud_rows":[0,64,63],"hud_cols":["a"]}')
    assert g["avatar"] is None and g["target"] is None
    assert g["click"] == {3}
    assert g["hud_rows"] == [0, 63] and g["hud_cols"] == []
