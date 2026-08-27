import numpy as np
from agents.causal.perception import parse, match_objects


def test_growing_object_keeps_id():
    g0 = np.zeros((8, 8), dtype=int); g0[2, 2] = 3; g0[2, 3] = 3   # 2 células
    s0 = match_objects(None, parse(g0.tolist()))
    id0 = s0.objects[0].id
    g1 = np.zeros((8, 8), dtype=int); g1[2, 2] = 3; g1[2, 3] = 3; g1[2, 4] = 3  # cresce p/ 3
    s1 = match_objects(s0, parse(g1.tolist()))
    assert len(s1.objects) == 1
    assert s1.objects[0].id == id0            # IoU = 2/3 >= 0.3 -> mantém id


def test_distinct_distant_objects_do_not_merge():
    g0 = np.zeros((10, 10), dtype=int); g0[1, 1] = 3
    s0 = match_objects(None, parse(g0.tolist()))
    g1 = np.zeros((10, 10), dtype=int); g1[1, 1] = 3; g1[8, 8] = 7   # novo objeto longe
    s1 = match_objects(s0, parse(g1.tolist()))
    ids = sorted(o.id for o in s1.objects)
    assert len(set(ids)) == 2                  # não funde


def test_recolor_plus_reshape_keeps_id_via_tier3():
    g0 = np.zeros((8, 8), dtype=int); g0[2, 2] = 3; g0[2, 3] = 3
    s0 = match_objects(None, parse(g0.tolist()))
    id0 = s0.objects[0].id
    g1 = np.zeros((8, 8), dtype=int); g1[2, 2] = 7; g1[2, 3] = 7; g1[2, 4] = 7  # cor nova + cresce
    s1 = match_objects(s0, parse(g1.tolist()))
    assert s1.objects[0].id == id0             # tier 3 (IoU qualquer cor)
