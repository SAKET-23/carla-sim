"""LiteSim ego: saturating-tyre bicycle model (simulation-backends.md section 1)."""

import math

import numpy as np

from hybrid_av.planning.route import Route
from hybrid_av.sim.base import Scenario, StepInfo
from hybrid_av.sim.litesim.actors import ActorManager
from hybrid_av.sim.litesim.rules import RuleChecker, light_state, surface_roughness
from hybrid_av.types import (
    ActuatorCommand,
    EgoState,
    Observation,
    TrafficContext,
    TrafficLightState,
)
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.dynamics import TAU_KIN, V_BLEND_LO, V_EPS, V_KIN_SWITCH
from hybrid_av.vehicle.params import VehicleParams

G = 9.81
TAU_ACC = 0.2  # s, first-order lag on acceleration
STEER_RATE = 3.0  # rad/s at the wheel
_SUBSTEPS = 5
_INF = math.inf


class LiteSim:
    def __init__(self, params: VehicleParams, lm: LongitudinalMap, dt: float = 0.05) -> None:
        self.p, self.lm, self.dt = params, lm, dt
        self._fzf = params.m * G * params.b / params.L
        self._fzr = params.m * G * params.a / params.L

    # --- Sim protocol -------------------------------------------------
    def route(self) -> Route:
        return self._route

    def close(self) -> None:
        pass

    def reset(self, scenario: Scenario) -> Observation:
        assert scenario.route is not None, "LiteSim needs scenario.route"
        self._scn, self._route = scenario, scenario.route
        r = self._route
        # state [vx, vy, r, X, Y, psi, delta, a]
        self._x = [
            scenario.ego_start_speed,
            0.0,
            0.0,
            float(r.x[0]),
            float(r.y[0]),
            float(r.psi[0]),
            0.0,
            0.0,
        ]
        self._t, self._s = 0.0, 0.0
        self._actors = ActorManager(scenario, r, np.random.default_rng(scenario.seed))
        self._rules = RuleChecker(scenario, r, self.p)
        self._objects = self._actors.step(0.0, 0.0, self.ego_state(), 0.0)
        return self._obs()

    def step(self, cmd: ActuatorCommand) -> tuple[Observation, StepInfo]:
        p, lm, dt = self.p, self.lm, self.dt
        x = self._x
        mu = self.mu_at(self._s)
        # wheel angle, rate limited
        d_cmd = -cmd.steer * p.max_steer_wheel
        x[6] += max(-STEER_RATE * dt, min(STEER_RATE * dt, d_cmd - x[6]))
        # invert the actuator map to a drive acceleration, clip to the traction limit
        if cmd.throttle > 0:
            req = (cmd.throttle - lm.throttle_bias) / lm.throttle_gain
        elif cmd.brake > 0:
            req = -cmd.brake / lm.brake_gain
        else:
            req = 0.0
        req = max(-mu * G, min(mu * G, req))
        h = dt / _SUBSTEPS
        for _ in range(_SUBSTEPS):
            k1 = self._f(x, mu, req)
            k2 = self._f([a + 0.5 * h * b for a, b in zip(x, k1)], mu, req)
            k3 = self._f([a + 0.5 * h * b for a, b in zip(x, k2)], mu, req)
            k4 = self._f([a + h * b for a, b in zip(x, k3)], mu, req)
            x = [a + h / 6.0 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
        if x[0] <= 0.0:  # brakes stop the car, they never reverse it
            x[0], x[7] = 0.0, max(x[7], 0.0)
            if cmd.throttle <= 0:
                x[1] = x[2] = 0.0
        self._x = x
        self._t += dt
        proj = self._route.project(x[3], x[4], s_hint=self._s)
        self._s = proj.s
        ego = self.ego_state()
        self._objects = self._actors.step(self._t - dt, dt, ego, self._s)
        info = self._rules.update(self._t, ego, self._s, proj.lateral, self._objects)
        return self._obs(), info

    # --- extras -------------------------------------------------------
    def ego_state(self) -> EgoState:
        vx, vy, r, X, Y, psi, _, a = self._x
        return EgoState(self._t, X, Y, psi, vx, vy, r, a)

    def steer_angle(self) -> float:
        return self._x[6]

    def mu_at(self, s: float) -> float:
        for z in self._scn.friction_zones:
            if z.s_start <= s <= z.s_end:
                return z.mu
        return self._scn.base_mu

    # --- internals ----------------------------------------------------
    def _obs(self) -> Observation:
        sc, t = self._scn, self._t
        front = self._s + self.p.a + 0.5
        d = self._route.sample([self._s])
        light, d_line = TrafficLightState.NONE, _INF
        for c in sc.controls:
            gap = c.s - front
            if c.kind == "light" and 0.0 <= gap <= 60.0 and gap < d_line:
                light, d_line = light_state(c, t), gap
        served = self._rules.served_stop_signs()
        d_sign = min(
            (c.s - front for i, c in enumerate(sc.controls)
             if c.kind == "stop_sign" and i not in served and 0.0 <= c.s - front <= 100.0),
            default=_INF,
        )  # fmt: skip
        d_junc = min((e - front for e, _ in sc.junction_s if 0.0 <= e - front <= 100.0), default=_INF)
        ctx = TrafficContext(
            light, d_line, d_sign, d_junc,
            float(d["speed_limit"][0]), float(d["lane_width"][0]), 0.0,
            surface_roughness(sc, self._s),
        )  # fmt: skip
        return Observation(self.ego_state(), self._objects, ctx, self.mu_at(self._s), "ground_truth")

    def _f(self, x: list[float], mu: float, req: float) -> list[float]:
        p = self.p
        vx, vy, r, psi, delta, a = x[0], x[1], x[2], x[5], x[6], x[7]
        c, s = math.cos(psi), math.sin(psi)
        xd, yd = vx * c - vy * s, vx * s + vy * c
        loss = p.rolling + p.drag_coeff * vx * vx if vx > 0 else 0.0
        ad = (req - loss - a) / TAU_ACC
        # kinematic
        r_kin = vx * math.tan(delta) / p.L
        kin = (a, (p.b * r_kin - vy) / TAU_KIN, (r_kin - r) / TAU_KIN)
        # dynamic with brush-style saturation
        v = max(vx, V_EPS)
        ff, fr = mu * self._fzf, mu * self._fzr
        fyf = ff * math.tanh(p.Cf * (delta - (vy + p.a * r) / v) / ff)
        fyr = fr * math.tanh(p.Cr * (-(vy - p.b * r) / v) / fr)
        cd = math.cos(delta)
        dyn = (a + r * vy, (fyf * cd + fyr) / p.m - r * vx, (p.a * fyf * cd - p.b * fyr) / p.Iz)
        w = max(0.0, min(1.0, (vx - V_BLEND_LO) / (V_KIN_SWITCH - V_BLEND_LO)))
        return [
            (1 - w) * kin[0] + w * dyn[0],
            (1 - w) * kin[1] + w * dyn[1],
            (1 - w) * kin[2] + w * dyn[2],
            xd, yd, r, 0.0, ad,
        ]  # fmt: skip
