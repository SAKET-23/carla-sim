"""LiteSim rule checker (simulation-backends.md section 1, Rules and infractions)."""

import math
from collections.abc import Sequence

import numpy as np

from hybrid_av.geometry import obb_overlap
from hybrid_av.planning.route import Route
from hybrid_av.sim.base import Scenario, StepInfo, TrafficControl
from hybrid_av.types import EgoState, ObjectClass, ObjectState, TrafficLightState
from hybrid_av.vehicle.params import VehicleParams

_BUMPER_MARGIN = 0.5  # m, front bumper = a + 0.5 ahead of the CG
_SERVE_DIST, _SERVE_SPEED = 3.0, 0.5  # state-vector.md: served = below 0.5 m/s within 3 m
_BLOCK_SPEED, _BLOCK_TIME, _RED_NEAR = 0.1, 180.0, 5.0
_DEVIATION = 30.0
_COMPLETE_MARGIN = 1.0
WET_ROUGHNESS = 0.8


def light_state(ctrl: TrafficControl, t: float) -> TrafficLightState:
    g, y, r = ctrl.cycle
    tau = (t - ctrl.phase_offset) % (g + y + r)
    if tau < g:
        return TrafficLightState.GREEN
    return TrafficLightState.YELLOW if tau < g + y else TrafficLightState.RED


# --- backend-independent checks, shared with CarlaSim (simulation-backends.md section 3) -------------------------


def crossed(prev_front: float | None, front: float, line_s: float) -> bool:
    """The front bumper moved across the stop line at route position line_s this step."""
    return prev_front is not None and prev_front < line_s <= front


class RouteProgress:
    """Monotone route completion: the furthest route position so far."""

    def __init__(self) -> None:
        self.s = 0.0

    def update(self, route_s: float) -> float:
        self.s = max(self.s, route_s)
        return self.s


class BlockedMonitor:
    """True once speed has stayed below 0.1 m/s for 180 s sim time, not counting time waiting at a red light."""

    def __init__(self) -> None:
        self._t_below: float | None = None

    def update(self, t: float, speed: float, at_red_light: bool) -> bool:
        if speed >= _BLOCK_SPEED or at_red_light:
            self._t_below = None
        elif self._t_below is None:
            self._t_below = t
        return self._t_below is not None and t - self._t_below >= _BLOCK_TIME


class StopSignMonitor:
    """Stop-sign rule: served below 0.5 m/s within 3 m of the line; crossing it unserved is a violation."""

    def __init__(self, controls: Sequence[TrafficControl]) -> None:
        self.controls, self.served = controls, set()

    def update(self, front: float, prev_front: float | None, speed: float) -> bool:
        violation = False
        for i, c in enumerate(self.controls):
            if c.kind != "stop_sign":
                continue
            if 0.0 <= c.s - front <= _SERVE_DIST and speed < _SERVE_SPEED:
                self.served.add(i)
            violation |= crossed(prev_front, front, c.s) and i not in self.served
        return violation


def deviation(lateral: float) -> bool:
    return abs(lateral) > _DEVIATION


def timeout(t: float, limit: float) -> bool:
    return t >= limit


def off_road_lateral(lateral: float, lane_width: float) -> bool:
    """Beyond the adjacent lane (the lateral half of the off-road rule)."""
    return abs(lateral) > 1.5 * lane_width


def route_completed(progress_s: float, route_length: float) -> bool:
    return progress_s >= route_length - _COMPLETE_MARGIN


def near_red_light(gap: float) -> bool:
    return 0.0 <= gap <= _RED_NEAR


def surface_roughness(sc: Scenario, route_s: float) -> float:
    """Surface-condition index: wet within 30 m before a low-mu zone and inside it."""
    wet = any(z.mu < 0.6 and z.s_start - 30.0 <= route_s <= z.s_end for z in sc.friction_zones)
    return WET_ROUGHNESS if wet else sc.base_roughness


class RuleChecker:
    def __init__(self, scenario: Scenario, route: Route, params: VehicleParams) -> None:
        self.sc, self.route, self.p = scenario, route, params
        self._parked = {a.id for a in scenario.actors if a.kind == "parked"}
        self.reset()

    def reset(self) -> None:
        self._stop = StopSignMonitor(self.sc.controls)
        self._prev_front: float | None = None
        self._blocked = BlockedMonitor()
        self._progress = RouteProgress()

    def served_stop_signs(self) -> set[int]:
        return set(self._stop.served)

    def _collision(self, ego: EgoState, objects: Sequence[ObjectState]) -> str:
        p = self.p
        off = p.a + _BUMPER_MARGIN - p.length / 2  # box centre relative to the CG
        box = (ego.x + off * math.cos(ego.psi), ego.y + off * math.sin(ego.psi), ego.psi, p.length, p.width)
        found = set()
        for o in objects:
            if obb_overlap(box, (o.x, o.y, o.yaw, o.length, o.width)):
                if o.id in self._parked:
                    found.add("static")
                elif o.cls == ObjectClass.PEDESTRIAN:
                    found.add("pedestrian")
                else:
                    found.add("vehicle")
        return next((c for c in ("pedestrian", "vehicle", "static") if c in found), "none")

    def update(
        self, t: float, ego: EgoState, ego_s: float, lateral: float, objects: Sequence[ObjectState]
    ) -> StepInfo:
        front = ego_s + self.p.a + _BUMPER_MARGIN
        prev = self._prev_front
        stop_violation = self._stop.update(front, prev, ego.vx)
        red_violation = red_near = False
        for c in self.sc.controls:
            if c.kind == "light" and light_state(c, t) == TrafficLightState.RED:
                red_near |= near_red_light(c.s - front)
                red_violation |= crossed(prev, front, c.s)
        self._prev_front = front

        blocked = self._blocked.update(t, ego.vx, red_near)
        route_s = self._progress.update(ego_s)
        lw = float(np.interp(ego_s, self.route.s, self.route.lane_width))
        return StepInfo(
            collision=self._collision(ego, objects),
            red_light_violation=red_violation,
            stop_sign_violation=stop_violation,
            off_road=off_road_lateral(lateral, lw),
            route_s=route_s,
            route_completed=route_completed(route_s, self.route.length),
            route_deviation=deviation(lateral),
            blocked=blocked,
            timeout=timeout(t, self.sc.time_limit),
            sim_time=t,
        )
