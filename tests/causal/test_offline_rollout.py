from analysis.offline_rollout import (
    is_deadlock, filter_trajectories, frequent_patterns, mdl_physics,
)


# --- §4: filtro de trajetórias em deadlock ---
def test_deadlock_same_action_repeated():
    assert is_deadlock(["A", "A", "A", "A", "A", "A"]) is True


def test_deadlock_short_cycle_repeated():
    assert is_deadlock(["A", "B", "A", "B", "A", "B"]) is True


def test_not_deadlock_when_score_gained():
    acts = ["A", "A", "A", "A", "A", "A"]
    scores = [0, 0, 0, 0, 0, 1]          # ganho na janela → não é deadlock
    assert is_deadlock(acts, scores) is False


def test_not_deadlock_varied_actions():
    assert is_deadlock(["A", "B", "C", "D", "E", "F"]) is False


def test_filter_drops_deadlock_episodes():
    eps = [
        {"actions": ["A", "A", "A", "A", "A", "A"]},          # deadlock → drop
        {"actions": ["A", "B", "C", "D", "E", "F"]},          # ok → keep
    ]
    kept = filter_trajectories(eps)
    assert len(kept) == 1
    assert kept[0]["actions"][0] == "A" and kept[0]["actions"][-1] == "F"


# --- §4: compressão MDL de padrões frequentes ---
def test_frequent_patterns_respects_min_support():
    trans = [("ACTION1", "moved")] * 4 + [("ACTION2", "none")] * 1
    pats = frequent_patterns(trans, min_support=3)
    assert len(pats) == 1
    assert pats[0]["pattern"] == ("ACTION1", "moved")
    assert pats[0]["support"] == 4


def test_frequent_patterns_ignores_exact_state():
    # (sig, key, effect) — deltas de estado ignorados, agrupa por (key, effect)
    trans = [("sigA", "ACTION1", "moved"), ("sigB", "ACTION1", "moved"),
             ("sigC", "ACTION1", "moved")]
    pats = frequent_patterns(trans, min_support=3)
    assert pats[0]["pattern"] == ("ACTION1", "moved")
    assert pats[0]["support"] == 3


def test_mdl_physics_keeps_only_productive():
    trans = [("ACTION1", "moved")] * 3 + [("ACTION2", "none")] * 5
    physics = mdl_physics(trans, min_support=3)
    assert physics == {"ACTION1": "moved"}     # 'none' descartado apesar do suporte
