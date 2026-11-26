import numpy as np
import pytest

from hybrid_av.eval.benchmark import _bootstrap_ci, run_benchmark
from hybrid_av.eval.metrics import ood_failed
from hybrid_av.eval.runner import EpisodeResult

KEYS = (
    "collision_pedestrian",
    "collision_vehicle",
    "collision_static",
    "red_light",
    "stop_sign",
    "off_road_m",
    "timeout",
    "blocked",
)


def _result(**kw):
    d = {
        "scenario": "x", "seed": 0, "route_completion": 1.0, "infractions": {k: 0 for k in KEYS},
        "driving_score": 100.0, "cross_track_rmse": 0.0, "heading_rmse_deg": 0.0, "steering_jerk_rms": 0.0,
        "mpc_solve_ms_p50": 1.0, "mpc_solve_ms_p99": 1.0, "mpc_fallbacks": 0, "slack_friction_steps": 0,
        "max_lat_accel_ratio": 0.5, "min_ttc": 10.0, "deadlock": False, "duration": 10.0, "trace": {},
    }  # fmt: skip
    d.update(kw)
    return EpisodeResult(**d)


def test_ood_failed_rules():
    assert ood_failed(_result(max_lat_accel_ratio=1.1), "wet_curve")
    assert ood_failed(_result(slack_friction_steps=11), "wet_curve")
    assert not ood_failed(_result(max_lat_accel_ratio=0.9, slack_friction_steps=10), "wet_curve")
    assert ood_failed(_result(deadlock=True), "merge_deadlock") and not ood_failed(
        _result(), "merge_deadlock"
    )
    assert ood_failed(_result(infractions={**{k: 0 for k in KEYS}, "off_road_m": 3}), "wet_curve")


def test_ood_failed_wet_curve_incomplete_route():
    assert ood_failed(_result(route_completion=0.6), "wet_curve")
    assert not ood_failed(_result(route_completion=1.0), "wet_curve")


def test_ood_failed_nominal_and_collision():
    assert not ood_failed(_result(), "nominal")
    assert ood_failed(_result(route_completion=0.9), "nominal")
    assert ood_failed(_result(infractions={**{k: 0 for k in KEYS}, "red_light": 1}), "nominal")
    assert ood_failed(_result(infractions={**{k: 0 for k in KEYS}, "collision_vehicle": 1}), "wet_curve")
    assert ood_failed(_result(route_completion=0.5), "merge_deadlock")


def test_bootstrap_ci_deterministic():
    x = np.arange(10.0)
    lo, hi = _bootstrap_ci(x)
    assert (lo, hi) == _bootstrap_ci(x) and lo < x.mean() < hi
    assert _bootstrap_ci(np.ones(5)) == (1.0, 1.0)


def test_benchmark_small(tmp_path):
    cfg = tmp_path / "b.yaml"
    cfg.write_text("backend: litesim\nmethods:\n  - {name: mpc_only, source: constant}\n"
                   "scenarios:\n  - {family: nominal, seeds: [50000, 2], overrides: {traffic: false}}\n")  # fmt: skip
    rep = run_benchmark(cfg, tmp_path / "out", workers=1)
    assert len(rep.rows) == 2 and rep.summary["mpc_only"]["nominal"]["route_completion"]["mean"] == 1.0
    assert {"rows.csv", "summary.json", "report.md"} <= {p.name for p in (tmp_path / "out").iterdir()}


def test_benchmark_rejects_carla(tmp_path):
    cfg = tmp_path / "b.yaml"
    cfg.write_text("backend: carla\nmethods: []\nscenarios: []\n")
    with pytest.raises(NotImplementedError):
        run_benchmark(cfg, tmp_path / "out")
