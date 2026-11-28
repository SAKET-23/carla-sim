"""IC-21...IC-23: friction zones (integration-check.md section E)."""

import importlib.util

import pytest
import yaml

from hybrid_av.sim.base import FrictionZone
from hybrid_av.sim.carla_backend.routes import plain_scenario, straight_route
from tests.integration.helpers import ROOT, Driver, spawn_tracked

pytestmark = pytest.mark.carla


def _wheels(vehicle) -> list[float]:
    return [w.tire_friction for w in vehicle.get_physics_control().wheels]


def _zone_run(make_sim, points, with_other=False):
    """Drive 5 m/s through FrictionZone(20..60, mu 0.35); yields (route_s, ego wheels, other wheels) per step."""
    import carla

    sim = make_sim()
    sc = plain_scenario(straight_route(points, "Town03_Opt", 150.0), 5.0, (FrictionZone(20.0, 60.0, 0.35),))
    obs = sim.reset(sc)
    other = None
    if with_other:
        spawn = sim.world.get_map().get_spawn_points()[0]
        other = spawn_tracked(sim, "vehicle.tesla.model3", carla.Transform(spawn.location, spawn.rotation))
    base = _wheels(sim.ego)
    other_base = _wheels(other) if other else None
    drv, rows = Driver(sim, 5.0), []
    for _ in range(400):
        obs, info = drv.step(obs)
        rows.append((info.route_s, _wheels(sim.ego), _wheels(other) if other else None))
        if info.route_s > 75.0:
            break
    return base, other_base, rows


def test_zone_applies_and_restores(make_sim, points):
    """IC-21 — see Plan/06-integration/integration-check.md; likely fix: ECC-16."""
    base, _, rows = _zone_run(make_sim, points)
    print(f"base tire_friction {base}")
    assert all(2.0 <= b <= 3.5 for b in base)
    inside = [w for s, w, _ in rows if 25.0 <= s <= 55.0]  # a tick after entering, a tick before leaving
    outside = [w for s, w, _ in rows if s > 66.0]
    assert inside and outside
    assert all(abs(f - 0.35 * b) < 1e-3 for w in inside for f, b in zip(w, base, strict=True))
    assert all(abs(f - b) < 1e-3 for f, b in zip(outside[-1], base, strict=True))


def test_other_vehicles_unchanged(make_sim, points):
    """IC-22 — see Plan/06-integration/integration-check.md; likely fix: ECC-16."""
    _, other_base, rows = _zone_run(make_sim, points, with_other=True)
    inside = [o for s, _, o in rows if 25.0 <= s <= 55.0]
    assert inside and all(o == pytest.approx(other_base, abs=1e-6) for o in inside)


def test_low_friction_limits_lateral_accel(make_sim):
    """IC-23 — see Plan/06-integration/integration-check.md; likely fix: ECC-16/ECC-17. Road-curve test (calibration.md §3)."""
    spec = importlib.util.spec_from_file_location(
        "calibrate_friction", ROOT / "scripts/calibrate_friction.py"
    )
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    from hybrid_av.vehicle.actuation import LongitudinalMap
    from hybrid_av.vehicle.params import VehicleParams

    sim = make_sim(no_rendering=True)
    rc = cal.parse_curve(cal.CURVE, "Town03_Opt")
    params = VehicleParams(max_steer_wheel=0.7)
    dry, _ = cal.curve_run(sim, rc, 1.0, params, LongitudinalMap())
    # the zone scale is the calibrated k* (calibration.md §3); before G2 Task 4 stores it, 0.35 (pod: 0.64 x dry)
    k = yaml.safe_load((ROOT / "configs/scenarios/wet_curve.yaml").read_text()).get(
        "carla_tire_friction_scale", 0.35
    )
    wet, _ = cal.curve_run(sim, rc, k, params, LongitudinalMap())
    print(f"a_lat_max dry {dry:.2f} wet {wet:.2f}")
    assert wet <= 0.6 * dry
