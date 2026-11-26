import numpy as np

from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import FrictionZone, Scenario


def build(seed: int, c: dict) -> Scenario:
    rng = np.random.default_rng(seed)
    arc = np.radians(c["angle_deg"])
    lim, lw = c["speed_limit"], c["lane_width"]
    route = RouteBuilder.concat(
        RouteBuilder.straight(c["straight_in"], lim, lw),
        RouteBuilder.arc(c["radius"], arc, lim, lw),
        RouteBuilder.straight(c["straight_out"], lim, lw),
    )
    # fixed draw order so {"dry": True} changes only mu (the roughness cue stays)
    mu, lead = float(rng.uniform(*c["mu"])), float(rng.uniform(*c["zone_lead"]))
    speed, rough = float(rng.uniform(*c["start_speed"])), float(rng.uniform(*c["roughness"]))
    zone = FrictionZone(
        c["straight_in"] - lead, c["straight_in"] + c["radius"] * arc, 1.0 if c["dry"] else mu
    )
    return Scenario(
        "wet_curve", seed, route, None, (zone,), 1.0, (), (), (),
        route.length / (0.5 * lim) + c["time_slack"], speed, rough,
    )  # fmt: skip
