# ruff: noqa: C408
import pytest

from hybrid_av.policy.io import policy_hash
from hybrid_av.weightspace.apply import apply_edit_plan, diff_policies
from hybrid_av.weightspace.schema import EditPlan, WeightEdit


def plan(*edits):
    return EditPlan(
        edits=[WeightEdit(rationale="r", **e) for e in edits], hypothesis="h", expected_effect="e"
    )


def test_set_add_scale_semantics(tiny_policy):
    p, d = apply_edit_plan(
        tiny_policy,
        plan(
            dict(tensor="fc1.weight", index=[0, 7], op="scale", value=3.0),
            dict(tensor="fc1.bias", index=[0], op="set", value=-0.25),
            dict(tensor="fc3.bias", index=[2], op="add", value=0.5),
        ),
    )
    assert p.fc1.weight[0, 7].item() == pytest.approx(3.0)
    assert p.fc1.bias[0].item() == pytest.approx(-0.25)
    assert p.fc3.bias[2].item() == pytest.approx(0.5)
    assert (d.n_changed, d.per_tensor) == (3, {"fc1.weight": 1, "fc1.bias": 1, "fc3.bias": 1})
    assert d.fraction_changed == pytest.approx(3 / 5955)
    assert d.max_abs_delta == pytest.approx(2.0)
    assert d.l2_delta == pytest.approx((2.0**2 + 0.25**2 + 0.5**2) ** 0.5)


def test_repeated_address_counts_once_and_applies_in_order(tiny_policy):
    p, d = apply_edit_plan(
        tiny_policy,
        plan(
            dict(tensor="fc1.bias", index=[5], op="set", value=1.0),
            dict(tensor="fc1.bias", index=[5], op="scale", value=3.0),
        ),
    )
    assert p.fc1.bias[5].item() == pytest.approx(3.0) and d.n_changed == 1


def test_apply_does_not_mutate_input(tiny_policy):
    h = policy_hash(tiny_policy)
    apply_edit_plan(tiny_policy, plan(dict(tensor="fc2.weight", index=[1, 1], op="set", value=9.0)))
    assert policy_hash(tiny_policy) == h


def test_noop_edit_counts_zero_changes(tiny_policy):
    noop = plan(dict(tensor="fc1.weight", index=[0, 7], op="set", value=1.0))
    assert apply_edit_plan(tiny_policy, noop)[1].n_changed == 0


def test_diff_policies_tolerance(tiny_policy):
    b, _ = apply_edit_plan(tiny_policy, plan(dict(tensor="fc1.weight", index=[1, 1], op="add", value=1e-3)))
    assert diff_policies(tiny_policy, b).n_changed == 1
    assert diff_policies(tiny_policy, b, tol=1e-2).n_changed == 0
