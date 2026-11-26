import numpy as np
import pytest

from hybrid_av.policy.io import policy_hash
from hybrid_av.policy.network import scale_action
from hybrid_av.weightspace import analysis as A
from hybrid_av.weightspace.schema import EditPlan, WeightEdit


def Z7(vals):
    Z = np.zeros((len(vals), 24), np.float32)
    Z[:, 7] = vals
    return Z


def test_activation_stats(tiny_policy):
    s = A.activation_stats(tiny_policy, Z7([-1.0, 0.5]))
    assert s["h1"]["active_frac"][0] == pytest.approx(0.5) and s["h1"]["max"][0] == pytest.approx(0.5)
    assert s["h1"]["dead"] == list(range(1, 64))
    assert s["h2"]["mean"][0] == pytest.approx(0.25)


def test_neuron_feature_attribution(tiny_policy):
    att = A.neuron_feature_attribution(tiny_policy, Z7([-1.0, 0.5]), layer="h1")
    assert att.shape == (64, 24) and att[0, 7] == pytest.approx(0.5) and np.count_nonzero(att) == 1


def test_output_attribution(tiny_policy):
    r = A.output_attribution(tiny_policy, Z7([0.5, 0.5]), output=0)
    assert r["contribution"][0] == pytest.approx(0.5) and np.count_nonzero(r["contribution"]) == 1
    assert r["bias"] == 0.0


def test_counterfactual(tiny_policy):
    out = A.counterfactual(
        tiny_policy, np.zeros(24, np.float32), feature=7, values=np.array([-1.0, 0.0, 1.0])
    )
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0, np.tanh(1.0)], atol=1e-6)


def test_evaluate_plan_offline_bias_shift(tiny_policy):
    edit = WeightEdit(tensor="fc3.bias", index=[0], op="add", value=-0.5, rationale="r")
    plan = EditPlan(edits=[edit], hypothesis="h", expected_effect="e")
    r = A.evaluate_plan_offline(tiny_policy, plan, Z7([0.5]), Z7([-1.0]))
    exp = (
        scale_action(np.array([np.tanh(0.0), 0, 0])).alpha_v
        - scale_action(np.array([np.tanh(0.5), 0, 0])).alpha_v
    )
    assert r["fail_mean_shift"]["alpha_v"] == pytest.approx(exp, abs=1e-6)
    assert r["fail_mean_shift"]["brake"] == pytest.approx(0.0, abs=1e-9) and r["diff"]["n_changed"] == 1


def test_analysis_never_changes_weights(tiny_policy):
    h = policy_hash(tiny_policy)
    A.neuron_feature_attribution(tiny_policy, Z7([0.5]), layer="h2")
    assert policy_hash(tiny_policy) == h
