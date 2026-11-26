"""IC-17...IC-20: route, speed limits, lane width, junctions and traffic-light context (section D)."""

import numpy as np
import pytest

from hybrid_av.rl.expert import ConstantAction
from hybrid_av.sim.carla_backend import convert
from hybrid_av.sim.carla_backend.client import CarlaSimConfig, connect
from hybrid_av.sim.carla_backend.routes import plain_scenario
from hybrid_av.sim.carla_backend.scenario_setup import chord_s, route_from_waypoints
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import TrafficLightState as Light
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from tests.integration.helpers import Driver, signalised_route, town03_pairs

pytestmark = pytest.mark.carla


@pytest.fixture(scope="module")
def pairs(carla_server):
    _, world = connect(CarlaSimConfig(town="Town03"))
    return town03_pairs(world)


def test_route_from_planner(make_sim, pairs):
    """IC-17 — see Plan/06-integration/integration-check.md; likely fix: ECC-13."""
    for start, goal, wps in pairs:
        route = route_from_waypoints(wps)
        assert np.all(np.diff(route.s) > 0) and np.max(np.diff(route.s)) <= 1.0
        assert np.max(np.abs(np.diff(route.psi))) < 0.3
        for wp in wps:
            x, y, _ = convert.loc_to_lib(wp.transform.location.x, wp.transform.location.y, 0.0)
            assert abs(route.project(x, y).lateral) < 0.05
        sim = make_sim()
        obs = sim.reset(plain_scenario({"town": "Town03", "start": start, "goal": goal}))
        assert abs(sim.route().project(obs.ego.x, obs.ego.y).lateral) < 0.5
        sim.close()


def _drive_constant_action(sim, sc, steps):
    obs = sim.reset(sc)
    stack = AutonomyStack(VehicleParams(max_steer_wheel=sim.max_steer), LongitudinalMap(), *_cfgs())
    stack.reset(sim.route())
    for _ in range(steps):
        feats = stack.features(obs)
        out = stack.act(obs, ConstantAction()(feats.z, feats), feats)
        yield obs, feats
        obs, info = sim.step(out.command)
        if info.route_completed:
            return


def _cfgs():
    from hybrid_av.mpc.controller import MPCConfig
    from hybrid_av.planning.reference import ReferenceConfig

    return MPCConfig(), ReferenceConfig()


def test_speed_limits(make_sim, pairs):
    """IC-18 — see Plan/06-integration/integration-check.md; likely fix: ECC-14."""
    start, goal, _ = pairs[0]
    sim = make_sim()
    sc = plain_scenario({"town": "Town03", "start": start, "goal": goal})
    agree = total = 0
    for _, feats in _drive_constant_action(sim, sc, 600):
        reported = sim.ego.get_speed_limit()  # km/h, 0 when unknown
        if reported:
            total += 1
            agree += abs(float(sim.route().sample([feats.route_s])["speed_limit"][0]) * 3.6 - reported) <= 1.0
    assert total > 50 and agree / total >= 0.95


def test_lane_width_and_junctions(make_sim, pairs):
    """IC-19 — see Plan/06-integration/integration-check.md; likely fix: ECC-14."""
    for start, goal, wps in pairs:
        sim = make_sim()
        sim.reset(plain_scenario({"town": "Town03", "start": start, "goal": goal}))
        s = chord_s(wps)
        width = sim.route().sample(s)["lane_width"]
        assert np.max(np.abs(width - [wp.lane_width for wp in wps])) < 0.05
        entries = [
            s[i] for i in range(len(wps)) if wps[i].is_junction and (i == 0 or not wps[i - 1].is_junction)
        ]
        assert len(entries) == len(sim._junctions)
        assert all(abs(a - b) <= 2.0 for a, b in zip(entries, sim._junctions, strict=True))
        sim.close()


def test_traffic_context_lights(make_sim, points):
    """IC-20 — see Plan/06-integration/integration-check.md; likely fix: ECC-15."""
    import carla

    sim = make_sim()
    _, world = connect(CarlaSimConfig(town="Town03"))
    rc, _, stop = signalised_route(world, points)
    obs = sim.reset(plain_scenario(rc, 6.0))
    drv = Driver(sim, 6.0)
    names = {
        carla.TrafficLightState.Red: Light.RED,
        carla.TrafficLightState.Yellow: Light.YELLOW,
        carla.TrafficLightState.Green: Light.GREEN,
    }
    last = np.inf
    for _ in range(400):
        ctx = obs.context
        if ctx.light_state != Light.NONE:
            assert ctx.dist_to_stop_line <= last + 1e-6, "dist_to_stop_line must not increase"
            last = ctx.dist_to_stop_line
            tf = sim.ego.get_transform().location
            geo = tf.distance(stop.transform.location) - sim._front
            assert abs(ctx.dist_to_stop_line - geo) < 1.0
        light = sim.ego.get_traffic_light()
        if light is not None:  # only inside the trigger volume
            assert ctx.light_state == names.get(light.get_state(), Light.NONE)
        if ctx.light_state != Light.NONE and ctx.dist_to_stop_line < 1.0:
            break
        obs, _ = drv.step(obs)
    assert last < 60.0, "never saw the light within 60 m"
