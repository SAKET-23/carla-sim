import os
import time

import numpy as np
import pytest

from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import FrictionZone, Scenario, Sim, TrafficControl
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.types import ActuatorCommand
from hybrid_av.types import TrafficLightState as L
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams


def _scn(route, mu=1.0, zones=(), v0=10.0):
    return Scenario("t", 0, route, None, zones, mu, (), (), (), 600.0, v0)


def test_is_sim():
    assert isinstance(LiteSim(VehicleParams(), LongitudinalMap()), Sim)


def test_context_fields():
    sc = Scenario(
        "t",
        0,
        RouteBuilder.straight(300, 13.89),
        None,
        (FrictionZone(150, 200, 0.35),),
        1.0,
        (),
        (TrafficControl("light", 40.0, (10, 3, 10), 0.0),),
        ((90.0, 100.0),),
        600,
        10,
        base_roughness=0.1,
    )
    obs = LiteSim(VehicleParams(), LongitudinalMap()).reset(sc)
    c = obs.context
    assert c.light_state == L.GREEN and abs(c.dist_to_stop_line - (40 - 1.7)) < 1e-6
    assert abs(c.dist_to_junction - (90 - 1.7)) < 1e-6
    assert c.dist_to_stop_sign == np.inf and c.surface_roughness == 0.1


def test_roughness_cue_before_wet_zone():
    sc = Scenario(
        "t",
        0,
        RouteBuilder.straight(300, 13.89),
        None,
        (FrictionZone(150, 200, 0.35),),
        1.0,
        (),
        (),
        (),
        600,
        10,
    )
    sim = LiteSim(VehicleParams(), LongitudinalMap())
    obs = sim.reset(sc)
    while obs.ego.x < 121.0:
        obs, _ = sim.step(ActuatorCommand(0.0, 0.3, 0.0))
    assert (
        obs.context.surface_roughness == 0.8 and obs.friction_true == 1.0
    )  # cue visible, friction still dry


@pytest.mark.benchmark
def test_speed_no_actors():
    sim = LiteSim(VehicleParams(), LongitudinalMap())
    sim.reset(_scn(RouteBuilder.straight(5000, 30)))
    t0 = time.perf_counter()
    for _ in range(2000):
        sim.step(ActuatorCommand(0.0, 0.3, 0.0))
    rate = 2000 / (time.perf_counter() - t0)
    print(f"LiteSim {rate:.0f} steps/s")
    if not os.environ.get("CI"):
        assert rate >= 2000
