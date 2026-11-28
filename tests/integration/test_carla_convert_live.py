"""IC-05...IC-10: coordinate conversions against a live server (integration-check.md section B)."""

import math

import numpy as np
import pytest

from hybrid_av.sim.carla_backend import convert
from hybrid_av.sim.carla_backend.routes import plain_scenario, straight_route
from tests.integration.helpers import RIG, Driver, straight_sim

pytestmark = pytest.mark.carla


def test_pose_matches_transform(make_sim, points):
    """IC-05 — see Plan/06-integration/integration-check.md; likely fix: ECC-04."""
    x, y, z, _ = points["straight"]
    rc = straight_route(points, "Town03_Opt") | {"start": [x, y, z, 30.0]}
    obs = make_sim().reset(plain_scenario(rc))
    assert obs.ego.x == pytest.approx(x, abs=1e-3)
    assert obs.ego.y == pytest.approx(-y, abs=1e-3)
    assert obs.ego.psi == pytest.approx(-math.radians(30.0), abs=1e-3)


@pytest.fixture(scope="module")
def left_turn(carla_server, points):
    """3 s at 8 m/s with library delta = +0.2: ego states, raw steer, raw angular velocity, converted IMU."""
    from hybrid_av.sim.carla_backend.client import CarlaSimConfig
    from hybrid_av.sim.carla_backend.sim import CarlaSim

    sim = CarlaSim(CarlaSimConfig(sensors=RIG))
    try:
        obs = sim.reset(plain_scenario(straight_route(points, "Town03_Opt"), 8.0))
        drv = Driver(sim, 8.0, 0.2)
        start, rec = obs.ego, {"ego": [], "steer": [], "wz_raw": [], "ay": [], "gz": []}
        for _ in range(60):
            cmd = drv.cmd(obs.ego)
            rec["ego"].append(obs.ego)
            rec["steer"].append(cmd.steer)
            obs, _ = sim.step(cmd)
            rec["wz_raw"].append(sim.ego.get_angular_velocity().z)
            imu = sim.last_sensor_data["imu"][-1]
            a, g = convert.imu_to_lib(imu.accelerometer, imu.gyroscope)
            rec["ay"].append(a[1])
            rec["gz"].append(g[2])
        rec["start"], rec["end"] = start, obs.ego
    finally:
        # closed before the tests run: a second live session in this process (make_sim) aborts in the Traffic
        # Manager when both switch it to synchronous mode on the same port (seen on the pod, G2)
        sim.close()
    return rec


def test_left_turn_signs(left_turn):
    """IC-06 — see Plan/06-integration/integration-check.md; likely fix: ECC-04/ECC-05."""
    egos, s = left_turn["ego"], left_turn["start"]
    assert all(c < 0 for c in left_turn["steer"])  # raw CARLA steer is negative for a left turn
    assert all(e.r > 0 for e in egos[10:])  # every step after 0.5 s
    assert left_turn["end"].psi - s.psi > 0.5
    dx, dy = left_turn["end"].x - s.x, left_turn["end"].y - s.y
    assert -math.sin(s.psi) * dx + math.cos(s.psi) * dy > 3.0  # y in the start frame


def test_body_velocities(make_sim, points, left_turn):
    """IC-07 — see Plan/06-integration/integration-check.md; likely fix: ECC-04."""
    sim, obs = straight_sim(make_sim, points, speed=10.0)
    drv = Driver(sim, 10.0)
    for _ in range(60):
        obs, _ = drv.step(obs)
    assert abs(obs.ego.vx - 10.0) < 0.3 and abs(obs.ego.vy) < 0.1
    # reported, not asserted: vy and r in the turn against the linear bicycle model's prediction
    e = left_turn["ego"][-1]
    print(f"turn: vy {e.vy:.3f} r {e.r:.3f} vx {e.vx:.2f}")


def test_imu_lateral_accel_sign(left_turn):
    """IC-08 — see Plan/06-integration/integration-check.md; likely fix: ECC-06."""
    ay = np.array(left_turn["ay"][30:])  # steady part, 1.5 s on
    vr = np.array([e.vx * e.r for e in left_turn["ego"][30:]])
    assert ay.mean() > 0 and ay.mean() == pytest.approx(vr.mean(), rel=0.2)


def test_imu_gyro_matches_angular_velocity(left_turn):
    """IC-09 — see Plan/06-integration/integration-check.md; likely fix: ECC-06."""
    gz = np.array(left_turn["gz"])
    truth = -np.radians(np.array(left_turn["wz_raw"]))
    moving = np.abs(truth) > 0.05
    assert moving.any() and np.all(np.sign(gz[moving]) == np.sign(truth[moving]))
    assert np.mean(np.abs(gz - truth)) < 0.02


def test_imu_rate(make_sim, points):
    """IC-10 — see Plan/06-integration/integration-check.md; likely fix: ECC-07."""
    sim, obs = straight_sim(make_sim, points, sensors=RIG)
    drv, counts = Driver(sim, 0.0), []
    for _ in range(50):
        obs, _ = drv.step(obs)
        counts.append(len(sim.last_sensor_data["imu"]))
    print(f"IMU samples per tick: {sorted(set(counts))}")  # record which; passes either way
    assert set(counts) <= {1, 5}  # the bridge publishes exactly what it receives (checked in IC-29)
