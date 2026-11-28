"""IC-24...IC-28: infractions and episode logic (integration-check.md section F)."""

import dataclasses
import math

import pytest

from hybrid_av.eval.driving_score import driving_score
from hybrid_av.eval.runner import run_episode
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.expert import ConstantAction
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.carla_backend.routes import carla_scenario, plain_scenario, straight_route
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import ActuatorCommand
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from tests.integration.helpers import ROUTES, Driver, ahead, signalised_route, spawn_tracked, straight_sim

pytestmark = pytest.mark.carla


@pytest.mark.parametrize(
    ("blueprint", "expected"),
    [
        ("vehicle.audi.tt", "vehicle"),  # a parked vehicle
        ("walker.pedestrian.0001", "pedestrian"),  # a static walker
        ("static.prop.streetbarrier", "static"),  # a wall-like static prop
    ],
)
def test_collisions(make_sim, points, blueprint, expected):
    """IC-24 — see Plan/06-integration/integration-check.md; likely fix: ECC-18."""
    import carla

    sim, obs = straight_sim(make_sim, points, speed=5.0)
    spawn_tracked(sim, blueprint, carla.Transform(ahead(points, 30.0), sim.ego.get_transform().rotation))
    drv, hit = Driver(sim, 5.0), None
    for _ in range(300):
        obs, info = drv.step(obs)
        if info.collision != "none":
            hit = info.collision
            break
    assert hit == expected


@pytest.mark.parametrize("colour", ["red", "green"])
def test_red_light(make_sim, points, colour):
    """IC-25 — see Plan/06-integration/integration-check.md; likely fix: ECC-15."""
    import carla

    sim = make_sim()
    rc, _, _ = signalised_route(carla.Client("localhost", 2000).get_world(), points)
    obs = sim.reset(plain_scenario(rc, 6.0))
    for lo in sim._lights:  # every light on the route is held in the chosen colour
        lo.light.set_state(carla.TrafficLightState.Red if colour == "red" else carla.TrafficLightState.Green)
        lo.light.freeze(True)
    drv, violations = Driver(sim, 6.0), 0
    for _ in range(500):
        obs, info = drv.step(obs)
        violations += info.red_light_violation
        if info.route_completed:
            break
    assert violations == (1 if colour == "red" else 0)


def test_off_road(make_sim, points):
    """IC-26 — see Plan/06-integration/integration-check.md; likely fix: ECC-19."""
    import carla

    sim, obs = straight_sim(make_sim, points, speed=5.0)
    drv, first_side, first_off = Driver(sim, 5.0), None, None
    for k in range(300):
        drv.delta = 0.0 if k < 40 else 0.15  # then steer left, off the road
        obs, info = drv.step(obs)
        wp = sim._wmap.get_waypoint(sim.ego.get_transform().location, project_to_road=False)
        side = wp is None or wp.lane_type != carla.LaneType.Driving
        first_side = k if side and first_side is None else first_side
        first_off = k if info.off_road and first_off is None else first_off
        if first_off is not None:
            break
    assert first_side is not None and first_off is not None and first_off - first_side <= 3
    sim.close()  # one live session at a time: the next ego spawns on the same spot (spawn collision on the pod)

    sim2, obs = straight_sim(make_sim, points, speed=5.0)  # adjacent lane: legal, not off-road
    drv = Driver(sim2, 5.0, 0.03)
    for _ in range(60):
        obs, info = drv.step(obs)
        wp = sim2._wmap.get_waypoint(sim2.ego.get_transform().location, project_to_road=False)
        if wp is not None and wp.lane_type == carla.LaneType.Driving:
            assert not info.off_road
    sim2.close()

    sim3, obs = straight_sim(make_sim, points)  # more than 30 m from the route
    tf = sim3.ego.get_transform()
    yaw = math.radians(tf.rotation.yaw)
    sim3.ego.set_location(
        carla.Location(
            tf.location.x - 35.0 * math.sin(yaw), tf.location.y + 35.0 * math.cos(yaw), tf.location.z + 0.5
        )
    )
    _, info = sim3.step(ActuatorCommand(0.0, 0.0, 1.0))
    assert info.route_deviation


@pytest.mark.slow
def test_blocked_timeout(make_sim, points):
    """IC-27 — see Plan/06-integration/integration-check.md; likely fix: ECC-19."""
    sim, _ = straight_sim(make_sim, points, no_rendering=True)
    first = None
    for k in range(1, 3701):
        _, info = sim.step(ActuatorCommand(0.0, 0.0, 1.0))
        if info.blocked and first is None:
            first = k
    assert first is not None and 3600 <= first <= 3610
    sim.close()  # its ego still occupies the spawn point sim2 uses (IC-27 spawn collision on the pod)
    sim2 = make_sim(no_rendering=True)  # a short time limit: timeout fires at t >= limit
    sim2.reset(dataclasses.replace(plain_scenario(straight_route(points, "Town03_Opt")), time_limit=5.0))
    infos = [sim2.step(ActuatorCommand(0.0, 0.0, 1.0))[1] for _ in range(110)]
    assert not infos[90].timeout and infos[105].timeout


def test_episode_result(make_sim):
    """IC-28 — see Plan/06-integration/integration-check.md."""
    try:
        sc = carla_scenario(make_scenario("nominal", 0), "nominal", 0, None, ROUTES)
    except LookupError:
        pytest.skip("no CARLA route for nominal: fill carla_routes.yaml (G2)")
    sim = make_sim(sensors=None)
    res = run_episode(
        sim,
        sc,
        AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig()),
        ConstantAction(),
    )
    fields = {
        k: v
        for k, v in dataclasses.asdict(res).items()
        if k not in ("trace", "infractions", "scenario", "seed", "deadlock")
    }
    assert all(math.isfinite(v) for k, v in fields.items() if k != "min_ttc")  # min_ttc is inf without a lead
    assert res.driving_score == pytest.approx(
        driving_score(res.route_completion, res.infractions, sim.route().length)
    )
