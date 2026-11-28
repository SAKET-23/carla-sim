"""Shared pieces of the live suite: driving helpers and tracked spawns."""

import math
from pathlib import Path

from hybrid_av.sim.carla_backend.routes import plain_scenario, straight_route
from hybrid_av.types import Control
from hybrid_av.vehicle.actuation import LongitudinalMap, control_to_actuator
from hybrid_av.vehicle.params import VehicleParams

ROOT = Path(__file__).resolve().parents[2]
RIG = ROOT / "configs/sensors/carla_rig.yaml"
ROUTES = ROOT / "configs/scenarios/carla_routes.yaml"
LM = LongitudinalMap()


def follow_steer(route, ego, max_steer: float, k_lat: float = 0.15, k_head: float = 1.0) -> float:
    """Library steering angle (left positive): steer right when left of the route, toward its heading."""
    p = route.project(ego.x, ego.y)
    err = math.atan2(math.sin(p.heading - ego.psi), math.cos(p.heading - ego.psi))
    return max(-max_steer, min(max_steer, k_head * err - k_lat * p.lateral))


class Driver:
    """Speed-hold PI on throttle/brake plus a fixed library steering angle (or, with follow=True, a P law
    on the planned route's lateral offset and heading error), through control_to_actuator."""

    def __init__(self, sim, v_ref: float, delta: float = 0.0, follow: bool = False) -> None:
        self.sim, self.v_ref, self.delta, self.follow, self.i = sim, v_ref, delta, follow, 0.0
        self.params = VehicleParams(max_steer_wheel=sim.max_steer)

    def cmd(self, ego):
        err = self.v_ref - ego.vx
        self.i += err * self.sim.dt
        delta = (
            follow_steer(self.sim.route(), ego, self.params.max_steer_wheel) if self.follow else self.delta
        )
        return control_to_actuator(Control(delta, 1.0 * err + 0.3 * self.i), ego.vx, self.params, LM)

    def step(self, obs):
        return self.sim.step(self.cmd(obs.ego))


def straight_sim(make_sim, points, speed=0.0, length=190.0, **cfg):
    """(sim, first observation) on the fixture's straight."""
    sim = make_sim(**cfg)
    rc = straight_route(points, "Town03_Opt", length)
    return sim, sim.reset(plain_scenario(rc, speed))


def ahead(points, distance: float, lateral: float = 0.0):
    """CARLA location `distance` m along the straight's heading and `lateral` m to its left (library convention)."""
    import carla

    x, y, z, yaw = points["straight"]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    # CARLA's left-handed frame: the driver's left is (+sin, -cos) of the yaw direction
    return carla.Location(x + distance * c + lateral * s, y + distance * s - lateral * c, z + 0.5)


def spawn_tracked(sim, blueprint: str, transform):
    """Spawn an extra actor that the sim's session destroys at close."""
    actor = sim.world.spawn_actor(sim.world.get_blueprint_library().find(blueprint), transform)
    sim._session.actors.append(actor)
    return actor


def town03_pairs(world, n: int = 5, seed: int = 0):
    """n start/goal pairs from spawn points whose planner route needs no lane change: [(start, goal, waypoints)]."""
    import random

    from hybrid_av.sim.carla_backend.scenario_setup import trace_waypoints

    rng, wmap, out = random.Random(seed), world.get_map(), []
    spawns = wmap.get_spawn_points()
    for _ in range(200):
        a, b = rng.sample(spawns, 2)
        start = [a.location.x, a.location.y, a.location.z, a.rotation.yaw]
        goal = [b.location.x, b.location.y, b.location.z]
        try:
            wps = trace_waypoints(wmap, start, goal)
        except ValueError:  # lane change in the trace
            continue
        if 150.0 < len(wps) < 600:  # resolution 1 m
            out.append((start, goal, wps))
        if len(out) == n:
            return out
    raise RuntimeError(f"found only {len(out)} lane-change-free pairs")


def route_gap(sim, ego_loc, stop_loc) -> float:
    """Arc length along the planned route from the ego's front bumper to a stop waypoint (the feature's definition)."""
    from hybrid_av.sim.carla_backend import convert

    route = sim.route()
    s_ego = route.project(*convert.loc_to_lib(ego_loc.x, ego_loc.y, 0.0)[:2]).s
    s_stop = route.project(*convert.loc_to_lib(stop_loc.x, stop_loc.y, 0.0)[:2]).s
    return s_stop - s_ego - sim._front


def signalised_route(world, points):
    """carla_route from the fixture's signalised-junction start through that junction, and the light's stop waypoint."""
    import carla

    start = points["signalised_junction"]
    for light in world.get_actors().filter("traffic.traffic_light*"):  # 0.9.16 has no get_traffic_lights
        for stop in light.get_stop_waypoints():
            back = stop.previous(90.0)
            if back and back[0].transform.location.distance(carla.Location(*start[:3])) < 5.0:
                goal = stop.next(30.0)[0].transform.location
                return {"town": "Town03_Opt", "start": start, "goal": [goal.x, goal.y, goal.z]}, light, stop
    raise RuntimeError("no traffic light matches the signalised_junction fixture")
