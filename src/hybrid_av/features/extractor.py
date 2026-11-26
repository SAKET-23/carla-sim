"""24-D feature extractor; the table is Plan/02-system-design/state-vector.md."""

from dataclasses import dataclass, replace

import numpy as np

from hybrid_av.features.friction import FrictionEstimator
from hybrid_av.features.spec import FEATURES, normalize
from hybrid_av.geometry import wrap_angle
from hybrid_av.planning.route import Route
from hybrid_av.types import Control, ObjectClass, ObjectState, Observation, TrafficLightState
from hybrid_av.vehicle.params import VehicleParams

BUMPER_EXTRA = 0.5  # m, front bumper beyond the front axle
LEAD_RANGE, PED_RANGE, CROSS_RANGE, STOP_RANGE, TTC_HI = 80.0, 50.0, 80.0, 100.0, 10.0
SAME_DIR = np.pi / 4  # heading difference below this: same direction; above: crossing
PED_FOV = np.pi / 3
CORRIDOR_MARGIN = 0.5  # m, added to ego half-width
CROSS_HORIZON = 10.0  # s
CLOSING_MIN = 0.1  # m/s
TAU_LEAD_ACCEL = 0.3  # s
JUNCTION_RANGE = 50.0  # m, minimum
JUNCTION_PLAN_DECEL = 1.0  # m/s², planning decel for the junction stop
JUNCTION_MARGIN = 10.0  # m
_VEHICLES = (ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.CYCLIST)


def junction_range(vx: float) -> float:
    """Distance within which the junction rule engages: the planned stopping distance plus a margin."""
    return max(JUNCTION_RANGE, vx**2 / (2 * JUNCTION_PLAN_DECEL) + JUNCTION_MARGIN)


@dataclass
class FeatureFrame:
    raw: np.ndarray  # (24,) physical units
    z: np.ndarray  # (24,) normalised [-1, 1], float32, the policy input
    route_s: float
    lateral: float
    heading_error: float
    mu_est: float
    lead: ObjectState | None
    stop_s: float | None
    dist_to_junction: float  # m along route to the next junction entry; inf if none within 100 m
    cross_dist_conflict: float  # m, ego front to the crossing conflict point; the runner's ttc_min gate


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


CROSS_MAX_ANGLE = np.radians(135.0)


class FeatureExtractor:
    def __init__(self, params: VehicleParams, route: Route) -> None:
        self.p = params
        self._fric = FrictionEstimator(params)
        self.reset(route)

    def reset(self, route: Route) -> None:
        self.route = route
        self.lateral_target = 0.0  # corridor centre, set by AutonomyStack.act
        self._fric.reset()
        self._s_hint: float | None = None
        self._lead_hist: tuple[int, float, float] | None = None  # id, along-route speed, filtered accel

    def extract(self, obs: Observation, last_control: Control, dt: float) -> FeatureFrame:
        p, route, ego, ctx = self.p, self.route, obs.ego, obs.context
        proj = route.project(ego.x, ego.y, s_hint=self._s_hint)
        self._s_hint = proj.s
        front_s = proj.s + p.a + BUMPER_EXTRA
        heading_error = wrap_angle(ego.psi - proj.heading)
        mu = self._fric.update(ego, last_control.delta, dt)

        look = max(20.0, 2.0 * ego.vx)
        ks = route.sample(np.minimum(np.arange(proj.s, proj.s + look + 0.25, 0.5), route.length))["kappa"]
        curvature = float(ks[np.argmax(np.abs(ks))])

        lead, lead_gap, lead_v, lead_lat = self._lead(obs, proj.s, front_s)
        lead_accel = self._lead_accel(lead, lead_v, dt)
        ped_dist, ped_bearing = self._pedestrian(obs, proj.heading)
        cross_dist, cross_speed, cross_ttc, cross_dist_conflict = self._crossing(obs, front_s)

        gap = TTC_HI  # intersection_gap
        if cross_ttc < TTC_HI and ctx.dist_to_junction <= junction_range(ego.vx):
            clear = (cross_dist_conflict + p.length + 2.0) / max(ego.vx, 2.0)
            gap = min(cross_ttc - clear, TTC_HI)

        stops = []  # distances to the stop line (yellow/red light) and the next unserved stop sign
        if ctx.light_state in (TrafficLightState.YELLOW, TrafficLightState.RED):
            stops.append(ctx.dist_to_stop_line)
        stops.append(ctx.dist_to_stop_sign)
        stops = [d for d in stops if np.isfinite(d) and d >= 0.0]
        stop_s = front_s + min(stops) if stops else None
        give_way = [ctx.dist_to_junction] if ctx.dist_to_junction <= junction_range(ego.vx) else []  # slot 23 only, never stop_s

        raw = np.array(
            [
                ego.vx, ego.vy, ego.r, proj.lateral, heading_error, curvature, mu,
                ctx.surface_roughness, ctx.road_grade,
                lead_gap, lead_v - ego.vx if lead else 0.0, self._lead_ttc(lead, lead_gap, lead_v, ego.vx),
                lead_accel, lead_lat if lead else 0.0,
                ped_dist, ped_bearing,
                cross_dist, cross_speed, cross_ttc,
                float(ctx.light_state), gap,
                ctx.speed_limit, ctx.lane_width, min([*stops, *give_way, STOP_RANGE]),
            ],
            dtype=np.float64,
        )  # fmt: skip
        assert raw.shape == (len(FEATURES),)
        return FeatureFrame(
            raw, normalize(raw).astype(np.float32), proj.s, proj.lateral, heading_error, mu, lead, stop_s,
            ctx.dist_to_junction, cross_dist_conflict,
        )  # fmt: skip

    def _project(self, o: ObjectState, ego_s: float, ego_x: float, ego_y: float, ego_psi: float):
        fwd = (o.x - ego_x) * np.cos(ego_psi) + (o.y - ego_y) * np.sin(ego_psi)
        pr = self.route.project(o.x, o.y, s_hint=ego_s + fwd)
        r = self.route
        if pr.s >= r.length - 1e-6:  # projection clamps at the end: keep the distance beyond it
            over = (o.x - r.x[-1]) * np.cos(r.psi[-1]) + (o.y - r.y[-1]) * np.sin(r.psi[-1])
            pr = replace(pr, s=pr.s + max(over, 0.0))
        return pr

    def _lead(self, obs: Observation, ego_s: float, front_s: float):
        """Nearest object in the corridor heading the same way; gap is bumper to its rear face."""
        e, best = obs.ego, None
        for o in obs.objects:
            pr = self._project(o, ego_s, e.x, e.y, e.psi)
            if abs(wrap_angle(o.yaw - pr.heading)) >= SAME_DIR:
                continue
            if abs(pr.lateral - self.lateral_target) > self.p.width / 2 + CORRIDOR_MARGIN:
                continue
            gap = pr.s - o.length / 2 - front_s
            if pr.s + o.length / 2 <= front_s or gap > LEAD_RANGE:
                continue  # entirely behind the bumper, or out of range
            if best is None or gap < best[1]:
                v_route = o.vx * np.cos(pr.heading) + o.vy * np.sin(pr.heading)
                best = (o, max(gap, 0.0), float(v_route), pr.lateral)
        return best if best else (None, LEAD_RANGE, 0.0, 0.0)

    @staticmethod
    def _lead_ttc(lead, gap, lead_v, ego_vx):
        closing = ego_vx - lead_v
        return min(gap / closing, TTC_HI) if lead and closing > CLOSING_MIN else TTC_HI

    def _lead_accel(self, lead, v, dt):
        if lead is None:
            self._lead_hist = None
            return 0.0
        a = 0.0
        if self._lead_hist and self._lead_hist[0] == lead.id and dt > 0:
            _, v0, a0 = self._lead_hist
            a = a0 + dt / (TAU_LEAD_ACCEL + dt) * ((v - v0) / dt - a0)
        self._lead_hist = (lead.id, v, a)
        return a

    def _pedestrian(self, obs: Observation, route_heading: float):
        e, best = obs.ego, (PED_RANGE, 0.0)
        for o in obs.objects:
            if o.cls != ObjectClass.PEDESTRIAN:
                continue
            dx, dy = o.x - e.x, o.y - e.y
            d = float(np.hypot(dx, dy))
            ang = np.arctan2(dy, dx)
            if d <= best[0] and (d == 0.0 or abs(wrap_angle(ang - route_heading)) <= PED_FOV):
                best = (d, wrap_angle(ang - e.psi))
        return best if best[0] < PED_RANGE else (PED_RANGE, 0.0)

    def _crossing(self, obs: Observation, front_s: float):
        """Vehicle reaching the conflict point soonest: (dist along its path, speed, ttc, route dist from bumper)."""
        none = (CROSS_RANGE, 0.0, TTC_HI, 0.0)
        route, e = self.route, obs.ego
        sg = np.arange(front_s, min(front_s + CROSS_RANGE, route.length), 0.5)
        if len(sg) < 2:
            return none
        rs = route.sample(sg)
        a = np.column_stack([rs["x"], rs["y"]])
        seg = a[1:] - a[:-1]
        best = None  # (t, dist, speed, s_conflict)
        for o in obs.objects:
            speed = float(np.hypot(o.vx, o.vy))
            if o.cls not in _VEHICLES or speed < CLOSING_MIN:
                continue
            pr = self._project(o, self._s_hint, e.x, e.y, e.psi)
            if abs(wrap_angle(o.yaw - pr.heading)) <= SAME_DIR:
                continue
            d = np.array([o.vx, o.vy]) * CROSS_HORIZON
            den = _cross(d, seg)
            ok = np.abs(den) > 1e-9
            w = a[:-1] - np.array([o.x, o.y])
            t = np.where(ok, _cross(w, seg) / np.where(ok, den, 1.0), np.inf)
            u = np.where(ok, _cross(w, d) / np.where(ok, den, 1.0), np.inf)
            hit = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
            if not hit.any():
                continue
            i = int(np.argmax(np.where(hit, -t, -np.inf)))  # smallest t
            if abs(wrap_angle(np.arctan2(o.vy, o.vx) - np.arctan2(seg[i, 1], seg[i, 0]))) >= CROSS_MAX_ANGLE:
                continue  # oncoming on a curve, not genuinely crossing
            tt, dist = t[i] * CROSS_HORIZON, t[i] * CROSS_HORIZON * speed
            if dist <= CROSS_RANGE and (best is None or tt < best[0]):
                best = (tt, dist, speed, sg[i] + u[i] * (sg[i + 1] - sg[i]) - front_s)
        if best is None:
            return none
        return best[1], best[2], min(best[0], TTC_HI), best[3]
