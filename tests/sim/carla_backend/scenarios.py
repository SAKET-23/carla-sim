"""Small Scenarios for the CarlaSim tests, built on the fake planner's straight line along CARLA +x."""

from hybrid_av.sim.base import Scenario


def straight_nominal(length: float = 100.0, **kw) -> Scenario:
    fields = {
        "name": "straight",
        "seed": 0,
        "route": None,
        "carla_route": {"town": "Town03_Opt", "start": [0.0, 0.0, 0.0, 0.0], "goal": [length, 0.0, 0.0]},
        "friction_zones": (),
        "base_mu": 1.0,
        "actors": (),
        "controls": (),
        "junction_s": (),
        "time_limit": 60.0,
        "ego_start_speed": 0.0,
    }
    return Scenario(**{**fields, **kw})
