import pytest

from hybrid_av.eval.runner import run_episode
from hybrid_av.features.friction import FrictionEstimator
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.expert import ConstantAction, ExpertPolicy
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams

EVAL = range(50000, 50050)  # development seeds (scenarios.md); never the 100,000+ benchmark range


def _run(family, seed, source, overrides=None, sim=None):
    sc = make_scenario(family, seed, overrides)
    return run_episode(
        sim or LiteSim(VehicleParams(), LongitudinalMap()),
        sc,
        AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig()),
        source,
    )


@pytest.mark.slow
def test_constant_action_traffic_free_nominal():
    for seed in list(EVAL)[:20]:
        r = _run("nominal", seed, ConstantAction(), {"traffic": False})
        assert r.route_completion == 1.0 and sum(r.infractions.values()) == 0, seed


@pytest.mark.slow
def test_expert_nominal_with_traffic():
    ok = {}
    for s in EVAL:
        r = _run("nominal", s, ExpertPolicy())
        ok[s] = (
            r.route_completion == 1.0
            and r.infractions["collision_pedestrian"]
            + r.infractions["collision_vehicle"]
            + r.infractions["collision_static"]
            == 0
        )
    assert [s for s in EVAL if not ok[s]] == []  # by seed
    assert sum(ok.values()) >= 48  # >= 95 % of 50


@pytest.mark.slow
def test_expert_merge_waits_without_collision():
    for seed in EVAL:
        r = _run("merge_deadlock", seed, ExpertPolicy())
        print(seed, r.route_completion, r.deadlock, r.infractions)
        assert r.infractions["collision_vehicle"] == 0, seed


@pytest.mark.slow
def test_wet_curve_slides_without_estimator(monkeypatch):
    monkeypatch.setattr(FrictionEstimator, "update", lambda self, ego, delta, dt: 1.0)
    slid = [_run("wet_curve", s, ConstantAction()).max_lat_accel_ratio > 1.0 for s in list(EVAL)[:20]]
    assert sum(slid) >= 16  # >= 80 %


@pytest.mark.slow
def test_wet_curve_dry_variant_does_not_slide():
    for s in list(EVAL)[:20]:
        assert _run("wet_curve", s, ConstantAction(), {"dry": True}).max_lat_accel_ratio < 1.0
