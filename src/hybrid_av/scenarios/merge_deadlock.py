import numpy as np

from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import Scenario, ScriptedActor
from hybrid_av.types import ObjectClass


def build(seed: int, c: dict) -> Scenario:
    rng = np.random.default_rng(seed)
    lim, lw = c["speed_limit"], c["lane_width"]
    route = RouteBuilder.concat(
        RouteBuilder.straight(c["straight_in"], lim, lw),
        RouteBuilder.arc(c["radius"], np.radians(c["angle_deg"]), lim, lw),
        RouteBuilder.straight(c["straight_out"], lim, lw),
    )
    # direction 0: a stream in each lane. Gaps are gap_min + Exp(gap_mean - gap_min) per lane, so
    # gap_mean 4.15 constrains the MERGED two-lane stream: gaps > 5 s recur about every 105 s (scenarios.md section 3).
    stream = ScriptedActor(1, ObjectClass.CAR, "cross_stream", {
        "junction_index": 0, "direction": 0, "speed": float(rng.uniform(*c["cross_speed"])),
        "gap_mean": c["gap_mean"], "gap_min": c["gap_min"], "path_offset": c["cross_path_offset"],
    })  # fmt: skip
    rough = float(rng.uniform(*c["roughness"]))
    return Scenario(
        "merge_deadlock", seed, route, None, (), 1.0, (stream,), (), (tuple(c["junction"]),),
        c["time_limit"], c["start_speed"], rough,
    )  # fmt: skip
