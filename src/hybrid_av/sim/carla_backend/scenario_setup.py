"""Route, traffic lights, friction zones and scripted actors for CarlaSim (simulation-backends.md §2, scenarios.md §3-4).

`carla` is imported lazily; the CARLA `agents` package comes from `$CARLA_ROOT/PythonAPI/carla` on PYTHONPATH.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np

from hybrid_av.planning.route import Route, RouteBuilder
from hybrid_av.sim.base import FrictionZone, Scenario
from hybrid_av.sim.carla_backend import convert
from hybrid_av.sim.litesim.actors import ActorManager
from hybrid_av.types import EgoState, ObjectClass

_DEFAULT_SPEED_LIMIT = 30.0 / 3.6  # m/s, town default when the route cache has no limits
_STOP_MATCH = 1.5  # m, a light's stop waypoint must lie this close to a route waypoint of the same lane
_START_SNAP = 5.0  # m, the planner's first waypoint must lie this close to the start pose
_CROSS_ID = 1000  # ActorManager numbers cross-stream vehicles 1000 * actor_id + n
_GROUND_CLEARANCE = 0.5  # m, spawn height above the road so a vehicle does not clip into it
_BLUEPRINT = {
    ObjectClass.CAR: "vehicle.audi.tt",
    ObjectClass.TRUCK: "vehicle.carlamotors.carlacola",
    ObjectClass.CYCLIST: "vehicle.diamondback.century",
    ObjectClass.PEDESTRIAN: "walker.pedestrian.0001",
}
_WHEEL_FIELDS = (
    "tire_friction", "damping_rate", "max_steer_angle", "radius", "max_brake_torque", "max_handbrake_torque",
    "lat_stiff_max_load", "lat_stiff_value", "long_stiff_value", "position",
)  # fmt: skip


class FrictionNotApplied(RuntimeError):
    def __init__(self, expected, got):
        super().__init__(f"tire_friction read-back {got} differs from the applied {expected}")
        self.expected, self.got = expected, got


# --- route ----------------------------------------------------------------------------------------------------


def trace_waypoints(world_map, start, goal, resolution: float = 1.0, planner=None) -> list:
    """GlobalRoutePlanner waypoints from start [x, y, z, yaw] to goal [x, y, z] (CARLA coordinates)."""
    import carla

    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        from agents.navigation.local_planner import RoadOption
    except ImportError as e:
        raise ImportError(
            "CARLA PythonAPI agents not on PYTHONPATH: set PYTHONPATH=$CARLA_ROOT/PythonAPI/carla"
        ) from e
    planner = planner or GlobalRoutePlanner(world_map, resolution)
    trace = planner.trace_route(carla.Location(*start[:3]), carla.Location(*goal[:3]))
    for i, (_, option) in enumerate(trace):
        # ECC-13: lane changes make lateral jumps in the route; choose start/goal pairs that need none
        if option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            raise ValueError(
                f"lane change in the route at waypoint {i}: choose a start/goal pair without one"
            )
    wps = []
    for wp, _ in trace:
        loc, fwd = wp.transform.location, math.radians(wp.transform.rotation.yaw)
        if wps:
            # IC-17: drop waypoints the planner emits behind the last kept one (waypoint-forward test); they would
            # make the polyline double back and the heading jump by pi
            prev = wps[-1].transform.location
            if (loc.x - prev.x) * math.cos(fwd) + (loc.y - prev.y) * math.sin(fwd) <= 0.0:
                continue
        wps.append(wp)
    # IC-37: the ego spawns at `start`; a route that begins elsewhere (other road, other lane) is never reached
    d = math.dist(start[:2], (wps[0].transform.location.x, wps[0].transform.location.y))
    if d > _START_SNAP:
        raise ValueError(
            f"route starts {d:.1f} m from the start pose {list(start[:3])}: choose another start"
        )
    return wps


def spawn_point_route(world_map, seed: int, length=(300.0, 600.0)) -> dict:
    """Fallback carla_route: spawn point 0 to a spawn point whose lane-change-free route is within `length` m,
    the seed picking among those candidates (deterministic)."""
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    spawns = world_map.get_spawn_points()
    start = spawns[0]
    planner = GlobalRoutePlanner(world_map, 1.0)
    candidates = []
    for sp in spawns[1:]:
        goal = [sp.location.x, sp.location.y, sp.location.z]
        try:
            wps = trace_waypoints(
                world_map, [start.location.x, start.location.y, start.location.z], goal, planner=planner
            )
        except ValueError:  # lane change in the route
            continue
        if length[0] <= chord_s(wps)[-1] <= length[1]:
            candidates.append(goal)
    if not candidates:
        raise LookupError(
            f"no spawn point {length[0]:.0f}-{length[1]:.0f} m from spawn point 0 on {world_map.name}"
        )
    s = start.location
    return {
        "town": world_map.name.split("/")[-1],
        "start": [s.x, s.y, s.z, start.rotation.yaw],
        "goal": candidates[seed % len(candidates)],
    }


def chord_s(wps) -> np.ndarray:
    """Arc length along the waypoints, library frame (equals Route.s up to resampling)."""
    xy = np.array([convert.loc_to_lib(w.transform.location.x, w.transform.location.y, 0.0)[:2] for w in wps])
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


def route_from_waypoints(wps, cache: dict | None = None) -> Route:
    # drop repeated points (the planner can emit zero-length steps); keep the lane width of the kept ones
    xy, lane_width = [], []
    for w in wps:
        p = convert.loc_to_lib(w.transform.location.x, w.transform.location.y, 0.0)[:2]
        if not xy or math.dist(p, xy[-1]) > 1e-3:
            xy.append(p)
            lane_width.append(w.lane_width)
    # ECC-14: speed limits come from the per-route cache (m/s, scalar or one per waypoint); town default otherwise
    limit = (cache or {}).get("speed_limit", _DEFAULT_SPEED_LIMIT)
    return RouteBuilder.from_waypoints(np.array(xy), limit, np.array(lane_width))


def build_route(world_map, start, goal, resolution: float = 1.0, cache: dict | None = None) -> Route:
    return route_from_waypoints(trace_waypoints(world_map, start, goal, resolution), cache)


@dataclass(frozen=True)
class LightOnRoute:
    light: object
    stop_s: float  # route arc length of the stop line


def route_lights(world, route_wps) -> list[LightOnRoute]:
    # ECC-15: ego.get_traffic_light() only works inside a trigger volume, so match every light's stop waypoints
    # (same road and lane, near a route waypoint) to the route up front
    s = chord_s(route_wps)
    xy = np.array([(w.transform.location.x, w.transform.location.y) for w in route_wps])
    found = []
    for light in (a for a in world.get_actors() if a.type_id.startswith("traffic.traffic_light")):
        best = None
        for stop in light.get_stop_waypoints():
            d = np.hypot(*(xy - (stop.transform.location.x, stop.transform.location.y)).T)
            for i in np.flatnonzero(d < _STOP_MATCH):
                same = (route_wps[i].road_id, route_wps[i].lane_id) == (stop.road_id, stop.lane_id)
                if same and (best is None or d[i] < best[0]):
                    best = (d[i], float(s[i]))
        if best is not None:
            found.append(LightOnRoute(light, best[1]))
    return sorted(found, key=lambda lo: lo.stop_s)


# --- friction -------------------------------------------------------------------------------------------------


class FrictionController:
    """Applies the zone's tire_friction scale to the ego, restores it outside, and checks the read-back."""

    def __init__(
        self, vehicle, zones: list[FrictionZone] | tuple[FrictionZone, ...], tire_scale: float | None = None
    ):
        self.vehicle, self.zones, self.tire_scale = vehicle, tuple(zones), tire_scale
        self.base = [w.tire_friction for w in vehicle.get_physics_control().wheels]
        self._zone: FrictionZone | None = None
        self._expected: list[float] | None = None

    def update(self, route_s: float) -> bool:
        zone = next((z for z in self.zones if z.s_start <= route_s <= z.s_end), None)
        if zone is self._zone:
            return False
        self._zone = zone
        # ECC-17: zone.mu is not 1:1 with tire_friction; the calibrated tire_scale replaces it when known
        scale = 1.0 if zone is None else (zone.mu if self.tire_scale is None else self.tire_scale)
        self._apply([b * scale for b in self.base])
        return True

    def _apply(self, frictions: list[float]) -> None:
        import carla

        physics = self.vehicle.get_physics_control()
        # ECC-16: new WheelPhysicsControl objects reassigned to .wheels (CARLA issue #5260); the other wheel fields
        # are carried over by the 0.9.16 constructor keyword names, which are unverified (ECC-40)
        physics.wheels = [
            carla.WheelPhysicsControl(**{**{f: getattr(w, f) for f in _WHEEL_FIELDS}, "tire_friction": f})
            for w, f in zip(physics.wheels, frictions, strict=True)
        ]
        self.vehicle.apply_physics_control(physics)
        self._expected = frictions

    def verify(self) -> None:
        """After the next tick: raise FrictionNotApplied if the vehicle does not report what was applied."""
        if self._expected is None:
            return
        got = [w.tire_friction for w in self.vehicle.get_physics_control().wheels]
        expected, self._expected = self._expected, None
        if any(abs(g - e) > 1e-3 for g, e in zip(got, expected, strict=True)):
            raise FrictionNotApplied(expected, got)


# --- scripted actors ------------------------------------------------------------------------------------------

_NO_EGO = EgoState(
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
)  # ActorManager.step ignores the ego (it uses ego_route_s)


class ActorDirector:
    """Places and drives the scenario's non-ego actors. Poses and gap schedules come from LiteSim's ActorManager,
    so cross traffic has exactly LiteSim's gap distribution (scenarios.md §3)."""

    def __init__(self, world, tm, scenario: Scenario, route: Route, rng, actors: list | None = None):
        self.world, self.tm, self.sc = world, tm, scenario
        self.actors = [] if actors is None else actors  # pass Session.actors so cleanup destroys everything
        self._mgr = ActorManager(scenario, route, rng)
        self._kind = {a.id: a for a in scenario.actors if a.kind != "cross_stream"}
        self._spawned: dict[int, object] = {}  # ObjectState id -> carla actor
        self._t = 0.0
        self._lead_speed: dict[int, float] = {}
        self.skipped: list[int] = []  # parked / pedestrian ids that failed to spawn

    def _place(self, o):
        import carla

        x, y, _ = convert.lib_to_loc(o.x, o.y, 0.0)
        wmap = self.world.get_map()
        ground = wmap.get_waypoint(carla.Location(x, y, 0.0)).transform.location.z
        transform = carla.Transform(
            carla.Location(x, y, ground + _GROUND_CLEARANCE), carla.Rotation(yaw=convert.lib_to_yaw(o.yaw))
        )
        return transform

    def _spawn_one(self, o, cls, try_only=False):
        import carla

        bp_lib = self.world.get_blueprint_library()
        transform = self._place(o)
        bp = bp_lib.find(_BLUEPRINT.get(cls, _BLUEPRINT[ObjectClass.CAR]))
        if try_only:
            actor = self.world.try_spawn_actor(bp, transform)
        else:
            # curb / map geometry can collide at the road edge: retry with the actor lifted 0, 0.5 and 1 m
            for lift in (0.0, 0.5, 1.0):
                lifted = carla.Transform(
                    carla.Location(transform.location.x, transform.location.y, transform.location.z + lift),
                    transform.rotation,
                )
                actor = self.world.try_spawn_actor(bp, lifted)
                if actor is not None:
                    break
        if actor is not None:
            self.actors.append(actor)
            self._spawned[o.id] = actor
        return actor

    def spawn(self) -> list:
        """Spawn lead, parked and pedestrian actors at their initial poses (cross-stream vehicles come in tick)."""
        for o in self._mgr.step(0.0, 0.0, _NO_EGO, 0.0):
            a = self._kind.get(o.id)
            if a is None:
                continue
            actor = self._spawn_one(o, a.cls)
            if actor is None:
                loc = self._place(o).location
                if a.kind == "lead":
                    raise RuntimeError(
                        f"lead actor {a.id} did not spawn at ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})"
                    )
                warnings.warn(f"{a.kind} actor {a.id} skipped: spawn collided", stacklevel=2)
                self.skipped.append(a.id)
                continue
            if a.kind == "lead":
                vehicle = actor
                vehicle.set_autopilot(True, self.tm.get_port())
                self.tm.ignore_lights_percentage(vehicle, 100.0)
                self.tm.distance_to_leading_vehicle(vehicle, 2.0)
                self.tm.auto_lane_change(vehicle, False)
                self._set_lead_speed(a.id, vehicle, a.params["v_cruise"])
        return [self._spawned[i] for i in self._kind if i in self._spawned]

    def _set_lead_speed(self, actor_id, vehicle, speed):
        if self._lead_speed.get(actor_id) != speed:
            self._lead_speed[actor_id] = speed
            self.tm.set_desired_speed(
                vehicle, speed * 3.6
            )  # ECC-39: TM takes km/h; LiteSim's lead brake profile is not enforced

    def tick(self, t: float, ego_route_s: float) -> None:
        import carla

        objs = self._mgr.step(self._t, t - self._t, _NO_EGO, ego_route_s)
        self._t = t
        live_cross = set()
        for o in objs:
            a = self._kind.get(o.id)
            actor = self._spawned.get(o.id)
            vx, vy, _ = convert.vel_to_lib(o.vx, o.vy, 0.0)  # the y flip is its own inverse: library -> CARLA
            if a is None:  # cross-stream vehicle
                live_cross.add(o.id)
                if actor is None:
                    # ECC-38: shortcut: cross vehicles are placed on the straight line the LiteSim layout implies; a curved
                    # cross road needs the lane's own waypoints (find_map_features, G2)
                    actor = self._spawn_one(o, o.cls, try_only=True)
                if actor is not None:
                    # drive it by velocity every tick, without autopilot, so it never yields
                    actor.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
            elif a.kind == "pedestrian_crossing":
                walker = actor
                speed = math.hypot(vx, vy)
                direction = carla.Vector3D(vx / speed, vy / speed, 0.0) if speed > 0 else carla.Vector3D()
                walker.apply_control(carla.WalkerControl(direction, speed))
            elif a.kind == "lead":
                p = a.params
                # LiteSim timing: the hold starts when brake_to is reached, then the lead recovers to cruise
                t_hold = p["brake_t"] + (p["v_cruise"] - p["brake_to"]) / p["brake_decel"]
                braking = 0 <= p["brake_t"] <= t < t_hold + p.get("brake_hold", math.inf)
                self._set_lead_speed(a.id, actor, p["brake_to"] if braking else p["v_cruise"])
        for oid in [i for i in self._spawned if i >= _CROSS_ID and i not in live_cross]:
            gone = self._spawned.pop(oid)  # passed the junction
            self.actors.remove(gone)
            gone.destroy()
