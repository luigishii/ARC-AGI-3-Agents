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


def test_direct_prompt_keyboard_no_click():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1", "ACTION2"]})
    assert '"type":"press"' in p
    assert '"type":"click_cell"' not in p


def test_direct_prompt_click_offers_click():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION6"]})
    assert '"type":"click_cell"' in p
    assert '"type":"press"' not in p


def test_direct_prompt_mixed_offers_both():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1", "ACTION6"]})
    assert '"type":"press"' in p and '"type":"click_cell"' in p


def test_direct_prompt_normalizes_action_names():
    p = build_direct_prompt(_scene(_obj()), {"available": ["GameAction.ACTION1"]})
    assert "ACTION1" in p
    assert "GameAction.ACTION1" not in p


def test_direct_prompt_hard_constrains():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION1"]})
    assert "ONLY from this list" in p


def test_direct_prompt_press_example_uses_available():
    p = build_direct_prompt(_scene(_obj()), {"available": ["ACTION3", "ACTION4"]})
    assert '{"type":"press","action":"ACTION3"}' in p
