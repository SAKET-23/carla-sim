# ruff: noqa: C408
import numpy as np
import torch

from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.stack import AutonomyStack
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from hybrid_av.weightspace.schema import EditPlan, WeightEdit, WeightSpaceConfig
from hybrid_av.weightspace.validate import EvalSummary, behavioural_gates, static_gates

CFG = WeightSpaceConfig()


def mk(*edits):
    return EditPlan(
        edits=[WeightEdit(rationale="r", **e) for e in edits], hypothesis="h", expected_effect="e"
    )


def by_name(rs):
    return {g.gate: g for g in rs}


def test_static_gates_pass_and_order(tiny_policy):
    g = by_name(
        static_gates(mk(dict(tensor="fc1.weight", index=[1, 1], op="set", value=2.0)), tiny_policy, CFG)
    )
    assert list(g) == ["count", "index_range", "value_range", "finite_outputs"] and all(
        r.passed for r in g.values()
    )


def test_count_gate_counts_changed_parameters(tiny_policy):
    many = mk(*[dict(tensor="fc2.weight", index=[i // 64, i % 64], op="set", value=0.5) for i in range(209)])
    assert not by_name(static_gates(many, tiny_policy, CFG))["count"].passed
    noop = mk(dict(tensor="fc1.weight", index=[0, 7], op="set", value=1.0))
    assert not by_name(static_gates(noop, tiny_policy, CFG))["count"].passed


def test_value_range_gate_uses_base_tensor_max(tiny_policy):
    # fc1.weight max |w| = 1.0 → limit 3.0
    ok = by_name(
        static_gates(mk(dict(tensor="fc1.weight", index=[2, 3], op="set", value=3.0)), tiny_policy, CFG)
    )
    bad = by_name(
        static_gates(mk(dict(tensor="fc1.weight", index=[2, 3], op="set", value=3.01)), tiny_policy, CFG)
    )
    assert (
        ok["value_range"].passed
        and not bad["value_range"].passed
        and "fc1.weight[2, 3]" in bad["value_range"].detail
    )


def test_finite_outputs_gate(seeded_policy):
    # value_range allows up to 3 × 2e38; the sum 2e38 + 3e38 exceeds float32 max (3.4e38) in h1_pre
    with torch.no_grad():
        seeded_policy.fc1.weight[0, 7] = 2e38
    plan = mk(dict(tensor="fc1.weight", index=[0, 8], op="set", value=3e38))
    g = by_name(static_gates(plan, seeded_policy, CFG, bundle_states=np.ones((4, 24), np.float32)))
    assert g["value_range"].passed and not g["finite_outputs"].passed


def test_behavioural_gates(tiny_policy, zero_policy):
    base = EvalSummary(0.1, {"lat_accel_ratio": 36}, np.array([90.0, 90.0]), frozenset({50003}), 2)
    good = EvalSummary(0.85, {"lat_accel_ratio": 6}, np.array([88.0, 89.0]), frozenset({50003}), 3)
    bad = EvalSummary(0.85, {}, np.array([80.0, 80.0]), frozenset({50003, 50007}), 5)
    table = {id(zero_policy): base, id(tiny_policy): good}
    g = by_name(behavioural_gates(tiny_policy, zero_policy, lambda p: table[id(p)], CFG))
    assert all(r.passed for r in g.values()) and list(g) == [
        "ood_success",
        "retention",
        "no_new_collisions",
        "mpc_fallbacks",
    ]
    table[id(tiny_policy)] = bad
    g = by_name(behavioural_gates(tiny_policy, zero_policy, lambda p: table[id(p)], CFG))
    assert not g["retention"].passed and not g["no_new_collisions"].passed and not g["mpc_fallbacks"].passed
    assert "50007" not in g["no_new_collisions"].detail


def test_feedback_is_aggregate_only():
    fb = EvalSummary(0.5, {"blocked": 20}, np.array([90.0, 70.0]), frozenset({50001}), 0).to_feedback()
    assert "50001" not in str(fb) and fb["nominal_mean_score"] == 80.0


def test_make_eval_fn_runs_gate_seeds_and_caches(zero_policy):
    import dataclasses

    from hybrid_av.eval.runner import EpisodeResult
    from hybrid_av.weightspace import validate

    calls = []

    def fake_episode(sim, scenario, stack, source):
        calls.append(scenario.seed if hasattr(scenario, "seed") else None)
        inf = {"collision_vehicle": 1 if len(calls) == 3 else 0, "off_road_m": 0}
        return EpisodeResult(
            "x", 0, 1.0, inf, 50.0, 0, 0, 0, 0, 0, 2, 0, 0.5, 9.0, False, 1.0, {"t": np.zeros(7)}
        )

    orig = validate.run_episode
    validate.run_episode = fake_episode
    try:
        cfg = dataclasses.replace(
            WeightSpaceConfig(), gate_ood_seeds=(50010, 50011), retention_seeds=(50000, 50000)
        )
        f = validate.make_eval_fn(
            "wet_curve",
            cfg,
            lambda: AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig()),
            workers=1,  # the monkeypatched run_episode only exists in this process
        )
        s = f(zero_policy)
        assert f(zero_policy) is s and len(calls) == 3
        assert (
            s.ood_success_rate == 1.0
            and s.nominal_mpc_fallbacks == 2
            and s.nominal_collision_seeds == frozenset({50000})
            and s.env_steps == 21
        )
    finally:
        validate.run_episode = orig


def test_fixtures_are_distinct(tiny_policy, zero_policy):
    assert tiny_policy is not zero_policy
    assert (
        float(zero_policy.fc1.weight[0, 7].detach()) == 0.0
        and float(tiny_policy.fc1.weight[0, 7].detach()) == 1.0
    )


def test_make_eval_fn_workers_do_not_change_results(seeded_policy):
    import dataclasses

    from hybrid_av.weightspace import validate

    cfg = dataclasses.replace(WeightSpaceConfig(), gate_ood_seeds=(0, 1), retention_seeds=(0, 1))
    out = [
        validate.make_eval_fn("nominal", cfg, validate.default_stack_factory, workers=w)(seeded_policy)
        for w in (1, 2)
    ]
    a, b = out
    assert a.env_steps > 0 and a.env_steps == b.env_steps
    assert a.ood_success_rate == b.ood_success_rate and a.nominal_mpc_fallbacks == b.nominal_mpc_fallbacks
    assert np.array_equal(a.nominal_scores, b.nominal_scores)
    assert a.nominal_collision_seeds == b.nominal_collision_seeds
