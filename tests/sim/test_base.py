import dataclasses

from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import FrictionZone, Scenario, Sim, StepInfo, TrafficControl


def test_scenario_construct_and_frozen():
    sc = Scenario(
        name="t",
        seed=1,
        route=RouteBuilder.straight(50, 10),
        carla_route=None,
        friction_zones=(FrictionZone(10, 20, 0.35),),
        base_mu=1.0,
        actors=(),
        controls=(TrafficControl("light", 30.0),),
        junction_s=(),
        time_limit=60.0,
        ego_start_speed=5.0,
    )
    assert sc.base_roughness == 0.0 and sc.controls[0].cycle == (10.0, 3.0, 10.0)
    assert [f.name for f in dataclasses.fields(StepInfo)][:2] == ["collision", "red_light_violation"]


def test_protocol_methods():
    assert {"reset", "step", "route", "close"} <= set(dir(Sim))
