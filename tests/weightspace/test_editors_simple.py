from hybrid_av.weightspace.editors.manual import ManualEditor
from hybrid_av.weightspace.editors.random_sparse import RandomSparseEditor
from hybrid_av.weightspace.schema import EditPlan, WeightEdit

REF = EditPlan(
    edits=[
        WeightEdit(tensor="fc1.weight", index=[i, 0], op="add", value=v, rationale="r")
        for i, v in enumerate([0.5, -1.0, 2.0, 0.5, -0.5])
    ],
    hypothesis="h",
    expected_effect="e",
)


class B:
    def __init__(self, p):
        self.policy = p


def test_manual_editor_reads_file(tmp_path, tiny_policy):
    f = tmp_path / "plan.json"
    f.write_text(REF.model_dump_json())
    ed = ManualEditor(f)
    assert ed.propose(B(tiny_policy), []) == REF and ed.usage == {"input_tokens": 0, "output_tokens": 0}


def test_random_sparse_matches_count_and_magnitudes(tiny_policy):
    p = RandomSparseEditor(REF, seed=0).propose(B(tiny_policy), [])
    assert len({(e.tensor, tuple(e.index)) for e in p.edits}) == 5 and {e.tensor for e in p.edits} == {
        "fc1.weight"
    }
    assert {abs(e.value) for e in p.edits} <= {0.5, 1.0, 2.0}


def test_random_sparse_seeded_and_redraws(tiny_policy):
    a, b = RandomSparseEditor(REF, seed=3), RandomSparseEditor(REF, seed=3)
    first = a.propose(B(tiny_policy), [])
    assert first == b.propose(B(tiny_policy), [])
    assert a.propose(B(tiny_policy), []) != first


def test_make_editor(tmp_path):
    import pytest

    from hybrid_av.weightspace.editors import make_editor
    from hybrid_av.weightspace.schema import WeightSpaceConfig

    cfg = WeightSpaceConfig()
    assert make_editor("random_sparse", cfg, reference_plan=REF, seed=1).name == "random_sparse"
    assert make_editor("manual", cfg, plan_path=tmp_path / "p.json").name == "manual"
    with pytest.raises(ValueError):
        make_editor("nope", cfg)
