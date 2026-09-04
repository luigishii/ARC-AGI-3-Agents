from types import SimpleNamespace

from agents.causal.llm import build_direct_prompt


def _obj(id=0, color=3):
    return SimpleNamespace(id=id, color=color, centroid=(1, 1), size=4, bbox=(1, 1, 2, 2))


def _scene(*objs):
    return SimpleNamespace(objects=list(objs))


def test_direct_prompt_lists_available():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1", "ACTION2"]})
    assert "AVAILABLE_ACTIONS" in p
    assert "ACTION1" in p and "ACTION2" in p


def test_direct_prompt_shows_objects():
    p = build_direct_prompt(_scene(_obj(id=0, color=3)), {"available": ["ACTION1"]})
    assert "OBJECTS" in p
    assert "color=3" in p


def test_direct_prompt_last_feedback_present():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1"]},
                            {"key": "ACTION2", "effect": "structural"})
    assert "ACTION2" in p and "structural" in p
    assert "PROGRESS" in p


def test_direct_prompt_last_omitted_when_none():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1"]}, None)
    assert "last action" not in p.lower()
    p2 = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1"]},
                             {"key": None, "effect": None})
    assert "last action" not in p2.lower()


def test_direct_prompt_asks_single_action():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1"]})
    assert '"type":"press"' in p
    assert '"type":"click_cell"' in p
