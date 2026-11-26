import numpy as np

from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import Scenario, ScriptedActor, TrafficControl
from hybrid_av.types import ObjectClass


def _route(rng: np.random.Generator, c: dict):
    lim, lw = float(rng.choice(c["speed_limits"])), c["lane_width"]
    while True:  # redraw the whole route from the same generator until the length fits
        segs = []
        for _ in range(int(rng.integers(c["n_segments"][0], c["n_segments"][1] + 1))):
            if rng.random() < 0.5:
                segs.append(RouteBuilder.straight(float(rng.uniform(*c["straight_length"])), lim, lw))
            else:
                ang = np.radians(rng.uniform(*c["arc_angle_deg"])) * rng.choice([-1.0, 1.0])
                segs.append(RouteBuilder.arc(float(rng.uniform(*c["arc_radius"])), float(ang), lim, lw))
        route = RouteBuilder.concat(*segs)
        if c["total_length"][0] <= route.length <= c["total_length"][1]:
            return route, lim


def build(seed: int, c: dict) -> Scenario:
    rng = np.random.default_rng(seed)
    route, lim = _route(rng, c)
    u = lambda key: float(rng.uniform(*c[key]))
    car, actors, controls, junctions = ObjectClass.CAR, [], [], []

    # one event per slot, so stop lines, junctions and crossings never overlap; every draw below is
    # unconditional so {"traffic": False} leaves the route and controls of a seed unchanged
    lo, hi = c["event_margin"][0], route.length - c["event_margin"][1]
    slots = rng.permutation(int((hi - lo) // c["event_slot"]))
    pos = [
        lo + float(k) * c["event_slot"] + float(rng.uniform(0.0, c["event_slot"] - c["junction_length"]))
        for k in slots[:4]
    ]
    has = rng.random(5) < [
        c["lead_prob"],
        c["light_prob"],
        c["stop_sign_prob"],
        c["junction_prob"],
        c["ped_prob"],
    ]

    s0, cruise_frac, brakes = u("lead_s0"), u("lead_cruise_frac"), rng.random() < c["lead_brake_prob"]
    brake_to, decel, brake_t = u("lead_brake_to_frac"), u("lead_brake_decel"), u("lead_brake_t")
    phase = float(rng.uniform(0.0, sum(c["light_cycle"])))
    cross_speed, direction = u("cross_speed"), float(rng.choice([-1.0, 1.0]))
    trigger, ped_speed, side = u("ped_trigger"), u("ped_speed"), float(rng.choice([-1.0, 1.0]))
    rough = u("roughness")
    brake_hold = u("lead_brake_hold")  # appended last so earlier draws keep their values

    if has[0]:
        v = cruise_frac * lim
        actors.append(ScriptedActor(1, car, "lead", {
            "s0": s0, "v_cruise": v, "brake_t": brake_t if brakes else -1.0,
            "brake_to": brake_to * v, "brake_decel": decel, "brake_hold": brake_hold,
        }))  # fmt: skip
    if has[1]:
        controls.append(TrafficControl("light", pos[0], tuple(c["light_cycle"]), phase))
    if has[2]:
        controls.append(TrafficControl("stop_sign", pos[1]))
    if has[3]:
        junctions.append((pos[2], pos[2] + c["junction_length"]))
        actors.append(ScriptedActor(2, car, "cross_stream", {
            "junction_index": 0, "direction": direction, "speed": cross_speed,
            "gap_mean": c["cross_gap_mean"], "gap_min": c["cross_gap_min"], "path_offset": c["cross_path_offset"],
        }))  # fmt: skip
    if has[4]:
        actors.append(ScriptedActor(3, ObjectClass.PEDESTRIAN, "pedestrian_crossing", {
            "s": pos[3], "trigger_dist": trigger, "speed": ped_speed, "side": side,
        }))  # fmt: skip
    return Scenario(
        "nominal", seed, route, None, (), 1.0, tuple(actors) if c["traffic"] else (), tuple(controls),
        tuple(junctions), route.length / (0.5 * lim) + c["time_slack"], c["start_speed_frac"] * lim, rough,
    )  # fmt: skip
