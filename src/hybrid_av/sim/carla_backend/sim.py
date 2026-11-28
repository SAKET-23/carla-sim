"""CarlaSim: the Sim backend for CARLA 0.9.16 (simulation-backends.md §2). `carla` is imported lazily."""

import dataclasses
import logging
import math
from contextlib import ExitStack
from typing import Any

import numpy as np

from hybrid_av.geometry import unwrap_near, wrap_angle
from hybrid_av.planning.route import Route
from hybrid_av.sim.base import Scenario, StepInfo
from hybrid_av.sim.carla_backend import convert
from hybrid_av.sim.carla_backend.client import CarlaSimConfig, carla_session, destroy_actors
from hybrid_av.sim.carla_backend.scenario_setup import (
    ActorDirector,
    FrictionController,
    chord_s,
    route_from_waypoints,
    route_lights,
    spawn_point_route,
    trace_waypoints,
)
from hybrid_av.sim.carla_backend.sensors import SensorQueues, load_rig, spawn_rig
from hybrid_av.sim.litesim.rules import (
    BlockedMonitor,
    RouteProgress,
    StopSignMonitor,
    crossed,
    deviation,
    light_state,
    near_red_light,
    off_road_lateral,
    route_completed,
    surface_roughness,
    timeout,
)
from hybrid_av.types import ActuatorCommand, Observation, TrafficContext
from hybrid_av.types import TrafficLightState as LibLight

_TIME_SLACK = 30.0  # s, nominal.yaml time_slack

log = logging.getLogger(__name__)
_SETTLE_TICKS = 10  # ECC-03: enough for the spawned ego to rest on the road
_SENSOR_TIMEOUT = 2.0  # s, wall clock, per frame (ECC-11)
_DEBOUNCE = 1.0  # s, one collision per actor id (ECC-18)
_INF = math.inf
_SPAWN_Z = 0.3  # m above the route start, so the ego does not spawn inside the road


def _collision_kind(type_id: str) -> str:
    # ECC-18: static map geometry reports an empty or "static.*" type id (or no actor at all)
    return (
        "pedestrian"
        if type_id.startswith("walker.")
        else "vehicle"
        if type_id.startswith("vehicle.")
        else "static"
    )


class CarlaSim:
    def __init__(self, cfg: CarlaSimConfig) -> None:
        self.cfg, self.dt = cfg, cfg.dt
        self._stack: ExitStack | None = None
        self._session = None
        self.last_sensor_data: dict[str, Any] = {}
        self.last_frame = -1
        self.max_steer = math.nan  # rad, read from the ego's wheels at reset

    # --- Sim protocol -------------------------------------------------
    def route(self) -> Route:
        return self._route

    def close(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            stack.close()  # the session destroys our actors (sensors stopped first) and restores the server

    def junction_intervals(self) -> list[tuple[float, float]]:
        """(entry_s, exit_s) of each junction along the driven route, for the steady-state metric mask."""
        return list(self._junction_intervals)

    def reset(self, scenario: Scenario) -> Observation:
        if scenario.carla_route is None:
            raise ValueError("CarlaSim needs scenario.carla_route")
        try:
            return self._reset(scenario, scenario.carla_route)
        except BaseException:
            self.close()  # never leave the server frozen in synchronous mode
            raise

    def step(self, cmd: ActuatorCommand) -> tuple[Observation, StepInfo]:
        import carla

        try:
            # ECC-05: ActuatorCommand.steer is already CARLA's right-positive [-1, 1]
            self.ego.apply_control(
                carla.VehicleControl(throttle=cmd.throttle, steer=cmd.steer, brake=cmd.brake)
            )
            self._tick()
            return self._observe(True)
        except BaseException:
            self.close()
            raise

    # --- reset --------------------------------------------------------
    def _reset(self, sc: Scenario, rc: dict) -> Observation:
        import carla

        cfg = self.cfg
        town = rc.get("town", cfg.town)
        if self._session is not None and not self._session.world.get_map().name.endswith(town):
            self.close()  # a new session loads the map and applies the settings after it (ECC-02)
        if self._session is None:
            self._stack = ExitStack()
            self._session = self._stack.enter_context(carla_session(dataclasses.replace(cfg, town=town)))
        else:
            destroy_actors(self._session.client, self._session.actors)
            self._session.actors.clear()
        s = self._session
        self.world, self.tm = s.world, s.tm
        wmap = self.world.get_map()
        self._sc, self._wmap = sc, wmap

        if rc.get("spawn_fallback"):
            rc = spawn_point_route(wmap, sc.seed)
            log.warning(
                "no route in carla_routes.yaml: using spawn point 0 %s -> %s on %s",
                rc["start"][:3],
                rc["goal"],
                rc["town"],
            )
        wps = trace_waypoints(wmap, rc["start"], rc["goal"])
        self._route = route_from_waypoints(wps, rc.get("cache"))
        if rc.get("time_from_route"):  # scenarios.md: time limit = length / (0.5 * speed limit) + slack
            lim = float(np.median(self._route.speed_limit))
            sc = dataclasses.replace(sc, time_limit=self._route.length / (0.5 * lim) + _TIME_SLACK)
            self._sc = sc
        self._wp_s = chord_s(wps)
        self._pitch = wrap_angle(
            np.radians([wp.transform.rotation.pitch for wp in wps])
        )  # CARLA reports 0..360, flat = ~360
        entry = [i for i, wp in enumerate(wps) if wp.is_junction and (i == 0 or not wps[i - 1].is_junction)]
        self._junctions = [float(self._wp_s[i]) for i in entry]
        # exit = first non-junction waypoint after the run (the route end if it ends inside one)
        self._junction_intervals = []
        for i in entry:
            j = next((k for k in range(i, len(wps)) if not wps[k].is_junction), len(wps) - 1)
            self._junction_intervals.append((float(self._wp_s[i]), float(self._wp_s[j])))

        bp_lib = self.world.get_blueprint_library()
        x, y, z, yaw = rc["start"]
        # a map spawn point sits up to ~1 m off the lane centre the planner route starts on; an off-road start
        # (the friction circle in the open area) stays as given
        if wmap.get_waypoint(
            carla.Location(x, y, z), project_to_road=False, lane_type=carla.LaneType.Driving
        ):
            lane = wmap.get_waypoint(
                carla.Location(x, y, z), project_to_road=True, lane_type=carla.LaneType.Driving
            ).transform
            x, y, z = lane.location.x, lane.location.y, lane.location.z
            # yaw too only when the start already points along the lane (IC-05 spawns 30 deg off on purpose)
            if abs((lane.rotation.yaw - yaw + 180.0) % 360.0 - 180.0) < 15.0:
                yaw = lane.rotation.yaw
        ego = self.world.spawn_actor(
            bp_lib.find(cfg.ego_blueprint),
            carla.Transform(carla.Location(x, y, z + _SPAWN_Z), carla.Rotation(yaw=yaw)),
        )
        s.actors.append(ego)
        self.ego = ego
        box = ego.bounding_box
        self._front = box.extent.x + box.location.x  # front bumper ahead of the actor origin
        # ECC-05: steer in [-1, 1] maps to the front wheels' max_steer_angle (degrees)
        phys = ego.get_physics_control()
        self.max_steer = math.radians(phys.wheels[0].max_steer_angle)
        v, k = convert.steer_curve_from_physics(
            phys
        )  # speed-dependent steering reduction, see control_to_actuator
        self.steer_params = {"max_steer_wheel": self.max_steer, "steer_curve_v": v, "steer_curve_k": k}
        self._fric = FrictionController(ego, sc.friction_zones, cfg.tire_friction_scale)
        wet = bool(sc.friction_zones)  # visual only, nothing depends on it
        self.world.set_weather(
            carla.WeatherParameters(wetness=80.0, precipitation_deposits=70.0)
            if wet
            else carla.WeatherParameters()
        )

        self._director = ActorDirector(
            self.world, self.tm, sc, self._route, np.random.default_rng(sc.seed), s.actors
        )
        self._director.spawn()
        collision = self.world.spawn_actor(
            bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=ego
        )
        s.actors.append(collision)
        self._events: list[tuple[int, int, str]] = []
        collision.listen(self._on_collision)
        self._queues = None
        if cfg.sensors is not None:
            specs = load_rig(cfg.sensors)
            self._queues = SensorQueues([sp.id for sp in specs])
            spawn_rig(s, ego, specs, self._queues)
        self._lights = route_lights(self.world, wps)
        self._set_lights(sc)

        self._fric.update(0.0)
        self._psi: float | None = None
        ego.apply_control(carla.VehicleControl())
        for _ in range(_SETTLE_TICKS):
            self._tick()
        # ECC-41: set_target_velocity along the heading gives LiteSim's ego_start_speed; get_velocity reads it back at once
        yaw = math.radians(ego.get_transform().rotation.yaw)
        v0 = sc.ego_start_speed
        ego.set_target_velocity(carla.Vector3D(v0 * math.cos(yaw), v0 * math.sin(yaw), 0.0))
        self._t0 = self._elapsed
        self._progress, self._blocked, self._stop = (
            RouteProgress(),
            BlockedMonitor(),
            StopSignMonitor(sc.controls),
        )
        self._prev_front: float | None = None
        self._last_hit: dict[int, float] = {}
        self._events.clear()
        self._s = 0.0
        return self._observe(False)[0]

    def _set_lights(self, sc: Scenario) -> None:
        import carla

        state = {
            LibLight.GREEN: carla.TrafficLightState.Green,
            LibLight.YELLOW: carla.TrafficLightState.Yellow,
        }
        for c in (c for c in sc.controls if c.kind == "light" and self._lights):
            light = min(self._lights, key=lambda lo: abs(lo.stop_s - c.s)).light
            light.set_green_time(c.cycle[0])
            light.set_yellow_time(c.cycle[1])
            light.set_red_time(c.cycle[2])
            # shortcut: initial colour from the cycle phase only; the elapsed time inside the phase is not set
            light.set_state(state.get(light_state(c, 0.0), carla.TrafficLightState.Red))

    # --- per tick -----------------------------------------------------
    def _on_collision(self, event) -> None:
        other = event.other_actor
        self._events.append((event.frame, getattr(other, "id", 0), getattr(other, "type_id", "")))

    def _tick(self) -> None:
        self.world.tick()
        snap = self.world.get_snapshot()
        self.last_frame, self._elapsed = snap.frame, snap.timestamp.elapsed_seconds
        if self._queues is not None:
            # ECC-07/11: data matched on this frame; a late or missing sensor gives None and a counter
            self.last_sensor_data = self._queues.collect(self.last_frame, _SENSOR_TIMEOUT)
        self._fric.verify()  # a friction change made last step must now read back

    def _unwrap(self, psi: float) -> float:
        self._psi = psi if self._psi is None else unwrap_near(psi, self._psi)
        return self._psi

    def _light_state(self, lo) -> LibLight:
        import carla

        names = {
            carla.TrafficLightState.Red: LibLight.RED,
            carla.TrafficLightState.Yellow: LibLight.YELLOW,
            carla.TrafficLightState.Green: LibLight.GREEN,
        }
        return names.get(lo.light.get_state(), LibLight.NONE)

    def _mu_at(self, s: float) -> float:
        return next((z.mu for z in self._sc.friction_zones if z.s_start <= s <= z.s_end), self._sc.base_mu)

    def _observe(self, stepped: bool) -> tuple[Observation, StepInfo]:
        import carla

        sc, ego, route = self._sc, self.ego, self._route
        t = self._elapsed - self._t0
        tf = ego.get_transform()
        state = convert.ego_from_actor(
            tf, ego.get_velocity(), ego.get_angular_velocity(), ego.get_acceleration(), t, self._unwrap
        )
        proj = route.project(state.x, state.y, s_hint=self._s if stepped else None)
        self._s = s = proj.s
        self._director.tick(t, s)
        front = s + self._front
        objects = tuple(self._object(actor) for actor in self._director.actors if self._is_traffic(actor))

        d = route.sample([s])
        light, d_line = LibLight.NONE, _INF
        for lo in self._lights:  # ECC-15: lights precomputed at reset, state queried every tick
            gap = lo.stop_s - front
            if 0.0 <= gap <= 60.0 and gap < d_line:
                light, d_line = self._light_state(lo), gap
        d_sign = min(
            (c.s - front for i, c in enumerate(sc.controls)
             if c.kind == "stop_sign" and i not in self._stop.served and 0.0 <= c.s - front <= 100.0),
            default=_INF,
        )  # fmt: skip
        d_junc = min((e - front for e in self._junctions if 0.0 <= e - front <= 100.0), default=_INF)
        ctx = TrafficContext(
            light, d_line, d_sign, d_junc,
            float(d["speed_limit"][0]), float(d["lane_width"][0]),
            float(np.interp(s, self._wp_s, self._pitch)),  # ECC-04: nose-up pitch = uphill, sign unverified
            surface_roughness(sc, s),
        )  # fmt: skip
        obs = Observation(state, objects, ctx, self._mu_at(s), "ground_truth")
        if not stepped:
            return obs, None  # type: ignore[return-value]

        prev = self._prev_front
        red = [lo for lo in self._lights if self._light_state(lo) == LibLight.RED]
        red_violation = any(crossed(prev, front, lo.stop_s) for lo in red)
        stop_violation = self._stop.update(front, prev, state.vx)
        self._prev_front = front
        blocked = self._blocked.update(t, state.vx, any(near_red_light(lo.stop_s - front) for lo in red))
        progress = self._progress.update(s)
        lw = float(d["lane_width"][0])
        # ECC-19: off-road = not on a Driving lane AND beyond the LiteSim lateral threshold
        wp = self._wmap.get_waypoint(tf.location, project_to_road=False)
        drivable = wp is not None and wp.lane_type == carla.LaneType.Driving
        info = StepInfo(
            collision=self._collisions(t),
            red_light_violation=red_violation,
            stop_sign_violation=stop_violation,
            off_road=not drivable and off_road_lateral(proj.lateral, lw),
            route_s=progress,
            route_completed=route_completed(progress, route.length),
            route_deviation=deviation(proj.lateral),
            blocked=blocked,
            timeout=timeout(t, sc.time_limit),
            sim_time=t,
        )
        self._fric.update(s)  # a change applies now and is read back after the next tick
        return obs, info

    def _is_traffic(self, actor) -> bool:
        return actor is not self.ego and actor.type_id.startswith(("vehicle.", "walker."))

    @staticmethod
    def _object(actor):
        box = actor.bounding_box
        wheels = int(actor.attributes.get("number_of_wheels", 4))
        return convert.object_from_actor(
            actor.id,
            actor.type_id,
            actor.get_transform(),
            box.extent,
            box.location,
            actor.get_velocity(),
            wheels,
        )

    def _collisions(self, t: float) -> str:
        events, self._events = self._events, []
        hits = set()
        for _, actor_id, type_id in events:
            if t - self._last_hit.get(actor_id, -_INF) >= _DEBOUNCE:  # ECC-18: one per actor id per second
                self._last_hit[actor_id] = t
                hits.add(_collision_kind(type_id))
        return next((c for c in ("pedestrian", "vehicle", "static") if c in hits), "none")
