"""features -> reference -> MPC -> actuator command (Plan/02-system-design/interfaces.md section 8)."""

from dataclasses import dataclass

import numpy as np

from hybrid_av.features.extractor import BUMPER_EXTRA, FeatureExtractor, FeatureFrame
from hybrid_av.mpc.controller import BicycleMPC, MPCConfig, MPCResult, Reference
from hybrid_av.planning.reference import ReferenceConfig, ReferenceGenerator
from hybrid_av.planning.route import Route
from hybrid_av.types import (
    ActuatorCommand,
    Control,
    ObjectState,
    Observation,
    PolicyAction,
    TrafficLightState,
)
from hybrid_av.vehicle.actuation import LongitudinalMap, control_to_actuator
from hybrid_av.vehicle.params import VehicleParams

_CORRIDOR_AHEAD = 80.0  # m
_CORRIDOR_MARGIN = 0.5  # m added to the ego half-width
HOLD_SPEED = 0.5  # m/s: below this a stop request holds the car (about one MPC step of braking from here)
HOLD_BRAKE = 0.5  # brake held at standstill


@dataclass
class StackOutput:
    command: ActuatorCommand
    control: Control
    action: PolicyAction
    features: FeatureFrame
    mpc: MPCResult
    reference: Reference


class AutonomyStack:
    def __init__(
        self, params: VehicleParams, lm: LongitudinalMap, mpc_cfg: MPCConfig, ref_cfg: ReferenceConfig
    ):
        self.params, self.lm, self.mpc_cfg, self.ref_cfg = params, lm, mpc_cfg, ref_cfg
        self.ref_gen = ReferenceGenerator(ref_cfg, mpc_cfg)
        self.mpc = BicycleMPC(params, mpc_cfg)
        self.last_control = Control(0.0, 0.0)

    def reset(self, route: Route) -> None:
        self.route = route
        if hasattr(self, "extractor"):
            self.extractor.reset(route)
        else:
            self.extractor = FeatureExtractor(self.params, route)
        self.mpc.reset()
        self.last_control = Control(0.0, 0.0)

    def features(self, obs: Observation) -> FeatureFrame:
        return self.extractor.extract(obs, self.last_control, self.mpc_cfg.dt)

    def act(
        self, obs: Observation, action: PolicyAction, features: FeatureFrame | None = None
    ) -> StackOutput:
        feats = features or self.features(obs)
        ego, ctx, dt = obs.ego, obs.context, self.mpc_cfg.dt
        stop_s = feats.stop_s
        if ctx.light_state == TrafficLightState.YELLOW and ctx.dist_to_stop_line < ego.vx**2 / (
            2 * self.ref_cfg.a_lon_comfort
        ):
            stop_s = None  # cannot stop at comfort deceleration: proceed through yellow
        if stop_s is not None:
            stop_s -= self.params.a + BUMPER_EXTRA  # line is a bumper position; the reference s is the CG's
        s0 = feats.route_s + ego.vx * dt  # aligns reference point 0 with the delay-compensated MPC state
        ref = self.ref_gen.generate(self.route, ego, s0, action, feats.mu_est, stop_s)
        res = self.mpc.solve(
            ego,
            ref,
            feats.mu_est,
            obstacles=self._obstacles(obs, feats.route_s, action.delta_y),
            brake_request=bool(action.brake >= 0.5),
        )
        command = control_to_actuator(res.control, ego.vx, self.params, self.lm)
        if (
            action.brake >= 0.5 and ego.vx < HOLD_SPEED
        ):  # stop request at standstill: the MPC would creep (R-30)
            command = ActuatorCommand(command.steer, 0.0, max(command.brake, HOLD_BRAKE))
        self.last_control = res.control
        self.extractor.lateral_target = action.delta_y
        return StackOutput(command, res.control, action, feats, res, ref)

    def _obstacles(self, obs: Observation, route_s: float, delta_y: float) -> list[ObjectState]:
        """Objects whose constant-velocity path enters the ego corridor within the horizon, nearest first."""
        r, c, ego = self.route, self.mpc_cfg, obs.ego
        i0, i1 = np.searchsorted(r.s, [route_s - 5.0, route_s + _CORRIDOR_AHEAD + 5.0])
        w = slice(max(i0 - 1, 0), i1 + 1)
        wx, wy, wpsi, ws = r.x[w], r.y[w], r.psi[w], r.s[w]
        t = np.arange(c.N + 1) * c.dt
        half = self.params.width / 2 + _CORRIDOR_MARGIN
        hits = []
        for o in obs.objects:
            px, py = o.x + o.vx * t, o.y + o.vy * t
            j = np.argmin((px[:, None] - wx) ** 2 + (py[:, None] - wy) ** 2, axis=1)
            lat = -(px - wx[j]) * np.sin(wpsi[j]) + (py - wy[j]) * np.cos(wpsi[j])
            inside = (
                (np.abs(lat - delta_y) <= half) & (ws[j] >= route_s) & (ws[j] <= route_s + _CORRIDOR_AHEAD)
            )
            if inside.any():
                hits.append((np.hypot(o.x - ego.x, o.y - ego.y), o))
        hits.sort(key=lambda h: h[0])
        return [o for _, o in hits[: c.max_obstacles]]
