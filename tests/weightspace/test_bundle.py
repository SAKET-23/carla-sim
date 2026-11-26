import json

import numpy as np
import pytest
import torch

from hybrid_av.eval.runner import EpisodeResult
from hybrid_av.features.spec import FEATURE_NAMES
from hybrid_av.policy.io import policy_hash, save_policy
from hybrid_av.policy.network import CompactPolicy
from hybrid_av.weightspace.bundle import classify_failure, export_bundle, load_bundle


def fake_result(seed, T, scenario="wet_curve", slack=0.0, fail_at=None):
    keys = [
        "t",
        "vx",
        "r",
        "route_s",
        "mpc_status",
        "slack_o",
        "alpha_v",
        "delta_y",
        "brake_flag",
        "infraction",
    ]
    tr = {k: np.zeros(T) for k in keys}
    tr["mu_true"] = np.ones(T)
    tr["slack_f"] = np.full(T, slack)
    tr["policy_step"] = np.arange(T) % 2 == 0
    tr["z"] = np.full((T, 24), 0.1, np.float32)
    if fail_at is not None:
        tr["vx"][fail_at:] = 10.0
        tr["r"][fail_at:] = 1.0  # |vx r| / (1 * 9.81) = 1.02 > 1.0
    return EpisodeResult(
        scenario=scenario,
        seed=seed,
        route_completion=0.4,
        infractions={},
        driving_score=40.0,
        cross_track_rmse=0.3,
        heading_rmse_deg=2.0,
        steering_jerk_rms=0.2,
        mpc_solve_ms_p50=3.0,
        mpc_solve_ms_p99=8.0,
        mpc_fallbacks=0,
        slack_friction_steps=0,
        max_lat_accel_ratio=1.02 if fail_at is not None else 0.0,
        min_ttc=10.0,
        deadlock=False,
        duration=T * 0.05,
        trace=tr,
    )


def test_classify_wet_curve_failure():
    assert classify_failure(fake_result(1, 40, fail_at=24), "wet_curve") == (True, "lat_accel_ratio", 24)
    assert classify_failure(fake_result(1, 40, slack=0.01), "wet_curve")[1] == "friction_slack"
    assert classify_failure(fake_result(1, 40), "wet_curve") == (False, "", -1)


def test_classify_collision_and_merge_reasons():
    r = fake_result(1, 40)
    r.infractions = {"collision_vehicle": 1}
    r.trace["infraction"][7:] = 2
    assert classify_failure(r, "wet_curve") == (True, "collision", 7)
    m = fake_result(2, 40, "merge_deadlock")
    assert classify_failure(m, "merge_deadlock") == (True, "incomplete", 39)
    m.infractions = {"blocked": 1}
    assert classify_failure(m, "merge_deadlock") == (True, "blocked", 39)
    m.route_completion = 1.0
    m.infractions = {}
    assert classify_failure(m, "merge_deadlock") == (False, "", -1)


def test_friction_slack_step_is_eleventh_slack_step():
    r = fake_result(1, 40)
    r.trace["slack_f"][5:] = 0.01
    assert classify_failure(r, "wet_curve") == (True, "friction_slack", 15)


def test_bundle_layout_and_roundtrip(tmp_path, tiny_policy):
    pp = tmp_path / "p.pt"
    save_policy(tiny_policy, pp, {"name": "tiny"})
    root = export_bundle(
        tiny_policy,
        [fake_result(50000, 40, fail_at=24), fake_result(50001, 30, fail_at=28)],
        [fake_result(0, 20, "nominal")],
        tmp_path,
        pp,
    )
    assert root.name == f"bundle_wet_curve_{policy_hash(tiny_policy)[:8]}"
    for f in [
        "README.md",
        "policy.pt",
        "policy.json",
        "feature_spec.json",
        "output_spec.json",
        "architecture.json",
        "scenario.json",
        "summary.json",
        "failures/seed_50000.npz",
        "failures/seed_50001.npz",
        "nominal/seed_0.npz",
    ]:
        assert (root / f).exists(), f
    tr = np.load(root / "failures/seed_50000.npz")
    assert tr["z"].shape == (20, 24) and tr["h1"].shape == (20, 64) and int(tr["failure_step"]) == 12
    b = load_bundle(root)
    assert b.policy_hash == policy_hash(tiny_policy) == policy_hash(b.policy)
    assert [f["name"] for f in b.feature_spec] == list(FEATURE_NAMES)
    assert b.Z_fail.shape == (35, 24) and b.Z_nom.shape == (10, 24) and b.Z_window.shape == (20, 24)
    assert json.loads((root / "architecture.json").read_text())["param_count"] == 5955


def test_bundle_trace_keys_and_summary(tmp_path, tiny_policy):
    pp = tmp_path / "p.pt"
    save_policy(tiny_policy, pp, {})
    root = export_bundle(tiny_policy, [fake_result(1, 40, fail_at=24)], [], tmp_path, pp)
    tr = np.load(root / "failures/seed_1.npz")
    keys = "z raw h1 h2 out_pre action route_s vx lat_accel_ratio mpc_status slack_friction slack_obstacle t"
    assert set(tr.files) == set(keys.split()) | {"outcome", "failure_step"}
    assert tr["action"].shape == (20, 3) and str(tr["outcome"]) == "lat_accel_ratio"
    b = load_bundle(root)
    assert (
        b.Z_nom.shape == (0, 24) and b.summary[0]["seed"] == 1 and b.summary[0]["reason"] == "lat_accel_ratio"
    )


def test_policy_json_reload_reproduces_outputs(tmp_path, seeded_policy):
    pp = tmp_path / "p.pt"
    save_policy(seeded_policy, pp, {})
    root = export_bundle(seeded_policy, [fake_result(1, 4, fail_at=2)], [], tmp_path, pp)
    sd = {k: torch.tensor(v) for k, v in json.loads((root / "policy.json").read_text()).items()}
    q = CompactPolicy()
    q.load_state_dict(sd)
    z = torch.rand(32, 24) * 2 - 1
    assert (q(z) - seeded_policy(z)).abs().max().item() < 1e-6


def test_bundle_rejects_mixed_scenarios(tmp_path, tiny_policy):
    pp = tmp_path / "p.pt"
    save_policy(tiny_policy, pp, {})
    with pytest.raises(ValueError, match="scenario"):
        export_bundle(
            tiny_policy,
            [fake_result(1, 4, "wet_curve"), fake_result(2, 4, "merge_deadlock")],
            [],
            tmp_path,
            pp,
        )


def test_load_bundle_detects_tampered_policy(tmp_path, tiny_policy):
    pp = tmp_path / "p.pt"
    save_policy(tiny_policy, pp, {})
    root = export_bundle(tiny_policy, [fake_result(1, 40, fail_at=24)], [], tmp_path, pp)
    with torch.no_grad():
        tiny_policy.fc1.bias[0] = 5.0
    save_policy(tiny_policy, root / "policy.pt", {})
    with pytest.raises(ValueError, match="hash"):
        load_bundle(root)
