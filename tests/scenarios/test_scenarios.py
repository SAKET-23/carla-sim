import numpy as np

from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.actors import ActorManager


def _same(a, b):  # Scenario holds numpy arrays (Route), so == on Scenario would raise
    return (
        np.array_equal(a.route.x, b.route.x)
        and np.array_equal(a.route.speed_limit, b.route.speed_limit)
        and (
            a.actors,
            a.controls,
            a.friction_zones,
            a.base_roughness,
            a.junction_s,
            a.time_limit,
            a.ego_start_speed,
        )
        == (
            b.actors,
            b.controls,
            b.friction_zones,
            b.base_roughness,
            b.junction_s,
            b.time_limit,
            b.ego_start_speed,
        )
    )


def test_deterministic():
    assert _same(make_scenario("nominal", 50001), make_scenario("nominal", 50001))
    assert not _same(make_scenario("nominal", 50001), make_scenario("nominal", 50002))


def test_nominal_ranges():
    for seed in range(50000, 50050):
        s = make_scenario("nominal", seed)
        assert 300 <= s.route.length <= 800 + 1 and s.base_mu == 1.0 and not s.friction_zones
        assert set(np.unique(s.route.speed_limit)) <= {8.33, 11.11, 13.89} and 0.0 <= s.base_roughness <= 0.3
    assert make_scenario("nominal", 50000, {"traffic": False}).actors == ()


def test_wet_curve_geometry():
    for seed in range(50000, 50020):
        s = make_scenario("wet_curve", seed)
        z = s.friction_zones[0]
        assert abs(np.max(s.route.kappa) - 1 / 30) < 1e-3 and 0.30 <= z.mu <= 0.40
        assert 100.0 <= z.s_start <= 110.0 and abs(z.s_end - (120 + 15 * np.pi)) < 0.5
        assert 12.0 <= s.ego_start_speed <= 14.0 and np.all(s.route.speed_limit == 13.89) and s.actors == ()
    assert make_scenario("wet_curve", 50000, {"dry": True}).friction_zones[0].mu == 1.0


def test_merge_deadlock():
    s = make_scenario("merge_deadlock", 50000)
    streams = [a for a in s.actors if a.kind == "cross_stream"]
    assert s.controls == () and len(s.junction_s) == 1 and s.time_limit == 240.0
    assert all(
        a.params["gap_mean"] == 4.15 and a.params["gap_min"] == 2.5 and a.params["direction"] == 0
        for a in streams
    )
    assert abs(np.max(s.route.kappa) - 1 / 12) < 2e-3


def test_merge_gap_statistics():
    # scenarios.md section 3: gaps above 5 s on the merged two-lane stream recur about once every 90-120 s (2026-11-18 tightening)
    s = make_scenario("merge_deadlock", 50000)
    am = ActorManager(s, s.route, np.random.default_rng(s.seed))
    lanes = [am.sample_gaps(i, 20000) for i in range(2)]
    assert all(g.min() >= 2.5 for g in lanes)
    merged_gaps = np.diff(np.sort(np.concatenate([np.cumsum(g) for g in lanes])))
    big = merged_gaps > 5.0
    assert 90.0 <= merged_gaps.sum() / big.sum() <= 120.0


def test_wet_curve_calibrated_tire_friction_scale():
    from pathlib import Path

    import yaml

    k = yaml.safe_load(Path("configs/scenarios/wet_curve.yaml").read_text())["carla_tire_friction_scale"]
    assert 0.0 < k < 1.0
