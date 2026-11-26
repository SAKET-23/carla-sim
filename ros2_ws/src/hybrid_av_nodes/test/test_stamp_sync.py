from hybrid_av_nodes.stamp_sync import StampSync


def test_stamp_sync_emits_only_complete_sets():
    s = StampSync(["ego", "objects", "context"])
    assert s.add("ego", 1.0, "e1") is None
    assert s.add("objects", 1.05, "o2") is None  # different tick
    assert s.add("objects", 1.0, "o1") is None
    assert s.add("context", 1.0, "c1") == {"ego": "e1", "objects": "o1", "context": "c1"}


def test_stamp_sync_bounded_backlog():
    s = StampSync(["a", "b"], max_pending=3)
    for k in range(10):
        s.add("a", k * 0.05, k)
    assert s.dropped == 7
    assert s.add("b", 0.0, "x") is None  # stamp 0.0 was dropped
    assert s.add("b", 0.45, "y") == {"a": 9, "b": "y"}
