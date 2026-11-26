"""LiteSim scripted actors; parameter keys in Plan/02-system-design/scenarios.md section 4."""

import copy
import math

import numpy as np

from hybrid_av.planning.route import Route
from hybrid_av.sim.base import Scenario, ScriptedActor
from hybrid_av.types import EgoState, ObjectState

_SPAWN_DIST = 80.0  # m before / after the conflict point
_LANE_GAP = 3.5  # m between the two lanes of a bidirectional stream
_CAR = (4.5, 1.9, 1.5)  # length, width, height
_PED = (0.5, 0.5, 1.7)
_PED_MARGIN = 2.0  # m beyond the lane edge where the pedestrian waits


class ActorManager:
    def __init__(self, scenario: Scenario, route: Route, rng: np.random.Generator) -> None:
        self.sc, self.route = scenario, route
        self._rng0 = copy.deepcopy(rng)
        self.reset()

    def _pose(self, s: float, lateral: float) -> tuple[float, float, float, float, float]:
        """World (x, y), route heading and left normal (nx, ny) at s, offset by `lateral`."""
        d = self.route.sample(np.array([s]))
        psi = float(d["psi"][0])
        nx, ny = -math.sin(psi), math.cos(psi)
        over = max(s - self.route.length, 0.0)  # past the route end: continue straight, do not park on it
        x, y = float(d["x"][0]) + over * math.cos(psi), float(d["y"][0]) + over * math.sin(psi)
        return x + lateral * nx, y + lateral * ny, psi, nx, ny

    def _streams(self, a: ScriptedActor) -> list[tuple[int, float]]:
        """(direction, route s of the lane) per lane; direction +1 moves along the left normal."""
        p = a.params
        s_c = self.sc.junction_s[int(p["junction_index"])][0] + p["path_offset"]
        d = int(p["direction"])
        # shortcut: lane layout for direction 0 (+1 lane behind, -1 lane ahead) is not in the docs
        return [(1, s_c - _LANE_GAP / 2), (-1, s_c + _LANE_GAP / 2)] if d == 0 else [(d, s_c)]

    def reset(self) -> None:
        self._rng = copy.deepcopy(self._rng0)
        self._lead = {}  # actor id -> [s, v, brake_started]
        self._ped = {}  # actor id -> [triggered, lateral]
        self._cross = []  # per lane: dict(actor, dir, s, speed, next_spawn, spawns [(t, n)], n)
        for a in self.sc.actors:
            if a.kind == "lead":
                self._lead[a.id] = [a.params["s0"], a.params["v_cruise"]]
            elif a.kind == "pedestrian_crossing":
                lw = float(self.route.sample(np.array([a.params["s"]]))["lane_width"][0])
                self._ped[a.id] = [False, a.params["side"] * (lw / 2 + _PED_MARGIN)]
            elif a.kind == "cross_stream":
                for d, s in self._streams(a):
                    lane = {"actor": a, "dir": d, "s": s, "spawns": [], "n": 0}
                    # pre-roll so the stream is already flowing at t = 0
                    lane["next"] = -2 * _SPAWN_DIST / a.params["speed"] + self._gap(a)
                    self._cross.append(lane)

    def _gap(self, a: ScriptedActor, rng: np.random.Generator | None = None) -> float:
        p = a.params
        return p["gap_min"] + (rng or self._rng).exponential(p["gap_mean"] - p["gap_min"])

    def sample_gaps(self, stream_index: int, n: int) -> np.ndarray:
        """Next n headways of a stream, from a separate generator (simulation unaffected)."""
        a = self._cross[stream_index]["actor"]
        rng = np.random.default_rng([self.sc.seed, 1, stream_index])
        return np.array([self._gap(a, rng) for _ in range(n)])

    def step(self, t: float, dt: float, ego: EgoState, ego_s: float) -> tuple[ObjectState, ...]:
        """Actor states at time t + dt."""
        te, out = t + dt, []
        lanes = iter(self._cross)
        for a in self.sc.actors:
            p = a.params
            if a.kind == "lead":
                st = self._lead[a.id]
                if p["brake_t"] >= 0 and t >= p["brake_t"]:
                    t_hold = p["brake_t"] + (p["v_cruise"] - p["brake_to"]) / p["brake_decel"]
                    if t >= t_hold + p.get("brake_hold", math.inf):
                        st[1] = min(st[1] + p.get("recover_accel", 1.5) * dt, p["v_cruise"])
                    else:
                        st[1] = max(st[1] - p["brake_decel"] * dt, p["brake_to"])
                st[0] += st[1] * dt
                x, y, psi, _, _ = self._pose(st[0], 0.0)
                out.append(
                    self._obj(a.id, a.cls, x, y, psi, _CAR, st[1] * math.cos(psi), st[1] * math.sin(psi))
                )
            elif a.kind == "pedestrian_crossing":
                st = self._ped[a.id]
                lw = float(self.route.sample(np.array([p["s"]]))["lane_width"][0])
                edge = lw / 2 + _PED_MARGIN
                st[0] = st[0] or p["s"] - ego_s <= p["trigger_dist"]
                v = 0.0
                if st[0] and st[1] * p["side"] > -edge + 1e-12:
                    step = min(p["speed"] * dt, st[1] * p["side"] + edge)
                    st[1] -= p["side"] * step
                    v = p["speed"]
                x, y, psi, nx, ny = self._pose(p["s"], st[1])
                out.append(self._obj(a.id, a.cls, x, y, psi, _PED, -p["side"] * v * nx, -p["side"] * v * ny))
            elif a.kind == "parked":
                x, y, psi, _, _ = self._pose(p["s"], p["lateral"])
                out.append(self._obj(a.id, a.cls, x, y, psi, (p["length"], p["width"], _CAR[2]), 0.0, 0.0))
            elif a.kind == "cross_stream":
                for _ in self._streams(a):
                    out.extend(self._cross_lane(next(lanes), te))
        return tuple(out)

    def _cross_lane(self, lane: dict, te: float) -> list[ObjectState]:
        a, v = lane["actor"], lane["actor"].params["speed"]
        while lane["next"] <= te:
            lane["spawns"].append((lane["next"], lane["n"]))
            lane["n"] += 1
            lane["next"] += self._gap(a)
        lane["spawns"] = [(ts, n) for ts, n in lane["spawns"] if v * (te - ts) - _SPAWN_DIST <= _SPAWN_DIST]
        d, res = lane["dir"], []
        for ts, n in lane["spawns"]:
            u = v * (te - ts) - _SPAWN_DIST  # distance past the conflict point along the travel direction
            x, y, psi, nx, ny = self._pose(lane["s"], d * u)
            res.append(
                self._obj(1000 * a.id + n, a.cls, x, y, psi + d * math.pi / 2, _CAR, d * v * nx, d * v * ny)
            )
        return res

    @staticmethod
    def _obj(id_, cls, x, y, yaw, dims, vx, vy) -> ObjectState:
        return ObjectState(id_, cls, x, y, 0.0, yaw, dims[0], dims[1], dims[2], vx, vy)
