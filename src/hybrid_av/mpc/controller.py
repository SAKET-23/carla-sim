import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import osqp

from hybrid_av.geometry import unwrap_near
from hybrid_av.mpc.constraints import friction_rows, obstacle_halfplanes
from hybrid_av.mpc.linearize import linearize_horizon
from hybrid_av.mpc.qp import QPBuilder
from hybrid_av.types import Control, EgoState, ObjectState
from hybrid_av.vehicle.dynamics import rk4_step
from hybrid_av.vehicle.params import VehicleParams

_NO_LIMIT = 1e10  # OSQP rejects time_limit <= 0; this is its own default, i.e. disabled


@dataclass(frozen=True, slots=True)
class MPCConfig:
    N: int = 20  # horizon steps
    dt: float = 0.05  # s
    Q: tuple[float, ...] = (1.0, 0.5, 0.5, 0.0, 0.0, 0.0)  # weights on [vx, vy, r, X, Y, psi]
    w_lat: float = 20.0  # cross-track error weight
    w_lon: float = 2.0  # along-track error weight
    w_psi: float = 10.0  # heading error weight
    R: tuple[float, float] = (1.0, 0.5)  # [delta, ax]
    R_delta: tuple[float, float] = (50.0, 1.0)  # per-step change of [delta, ax]
    P_scale: float = 5.0  # terminal weight = P_scale * stage weight
    delta_max: float = 0.70  # rad
    ddelta_max: float = 0.15  # rad per step
    ax_min: float = -4.5
    ax_max: float = 2.5
    mu_default: float = 1.0
    friction_margin: float = 0.85  # use 85 % of mu*g
    w_slack_friction: float = 1e4
    w_slack_obstacle: float = 1e5
    max_obstacles: int = 4
    obstacle_margin: float = 1.0  # m added to half-widths
    max_iter: int = 4000
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    time_limit: float = 0.0  # s, OSQP wall-clock limit; 0 = off (deterministic; max_iter bounds the work), see configs/mpc/realtime.yaml
    cold_start_time_factor: float = (
        4.0  # first solve after reset() gets this x time_limit (only if time_limit > 0)
    )


@dataclass(frozen=True, slots=True)
class Reference:
    """Horizon reference. All arrays have length N+1 (k = 0..N)."""

    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray  # unwrapped and continuous with ego heading
    v: np.ndarray
    kappa: np.ndarray  # curvature, 1/m
    s: np.ndarray  # route arc length, m


@dataclass(frozen=True, slots=True)
class MPCResult:
    """On `fallback_*` statuses, `cost`, `slack_friction_max` and `slack_obstacle_max` are 0.0 and not meaningful; read `status` first."""

    control: Control  # first move, after any fallback
    status: Literal["solved", "solved_inaccurate", "fallback_shift", "fallback_brake"]
    solve_time: float  # s, wall clock including assembly
    predicted: np.ndarray  # (N+1, 6)
    slack_friction_max: float
    slack_obstacle_max: float
    cost: float


class BicycleMPC:
    def __init__(self, params: VehicleParams, cfg: MPCConfig) -> None:
        self.params, self.cfg = params, cfg
        self.qp = QPBuilder(params, cfg)
        self.reset()

    def reset(self) -> None:
        self._prob: osqp.OSQP | None = (
            None  # OSQP keeps duals/rho across solves; a new episode must start from a fresh setup
        )
        self._u_applied = np.zeros(2)
        self._plan_x: np.ndarray | None = None  # previous (N+1, 6) prediction
        self._plan_u: np.ndarray | None = None  # previous (N, 2) inputs
        self._fails = 0
        self._cost = 0.0
        self._status = "solved"

    def _nominal(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Previous plan shifted one step (last input repeated), or a zero-input rollout on the first call."""
        N, p, dt = self.cfg.N, self.params, self.cfg.dt
        if self._plan_x is None:
            u_bar = np.zeros((N, 2))
            x_bar = np.empty((N + 1, 6))
            x_bar[0] = x0
            for k in range(N):
                x_bar[k + 1] = rk4_step(x_bar[k], u_bar[k], p, dt)
            return x_bar, u_bar
        u_bar = np.vstack([self._plan_u[1:], self._plan_u[-1:]])
        x_bar = np.vstack([self._plan_x[1:], rk4_step(self._plan_x[-1], u_bar[-1], p, dt)])
        x_bar[:, 5] += unwrap_near(x_bar[0, 5], x0[5]) - x_bar[0, 5]
        x_bar[0] = x0
        return x_bar, u_bar

    def _solve_qp(self, Pm, q, Acon, l, u, z0):
        """Returns the solution vector, or None unless OSQP reports solved / solved inaccurate."""
        c = self.cfg
        if self._prob is None:
            self._prob = osqp.OSQP()
            self._prob.setup(
                Pm, q, Acon, l, u, eps_abs=c.eps_abs, eps_rel=c.eps_rel, max_iter=c.max_iter,
                time_limit=c.time_limit or _NO_LIMIT, polishing=False, verbose=False, warm_starting=True,
            )  # fmt: skip
        else:
            self._prob.update(Px=Pm.data, q=q, Ax=Acon.data, l=l, u=u)
        self._prob.warm_start(x=z0)
        cold = (
            c.time_limit > 0 and self._plan_x is None
        )  # no previous plan: the nominal guess is far from the solution
        if cold:
            self._prob.update_settings(time_limit=c.time_limit * c.cold_start_time_factor)
        res = self._prob.solve(raise_error=False)
        if cold:
            self._prob.update_settings(time_limit=c.time_limit)
        if res.info.status_val not in (1, 2) or not np.all(np.isfinite(res.x)):
            return None
        self._status = "solved" if res.info.status_val == 1 else "solved_inaccurate"
        self._cost = float(res.info.obj_val)
        return res.x

    def solve(
        self,
        ego: EgoState,
        ref: Reference,
        mu: float,
        obstacles: Sequence[ObjectState] = (),
        brake_request: bool = False,
    ) -> MPCResult:
        t0 = time.perf_counter()
        c, p, qp = self.cfg, self.params, self.qp
        N = c.N
        mu = max(mu, 1e-3)  # a non-positive estimate would make the a_x bounds cross
        x0 = rk4_step(ego.as_mpc_vector(), self._u_applied, p, c.dt)  # one-step delay compensation
        shift = unwrap_near(ref.psi[0], x0[5]) - ref.psi[0]
        ref = Reference(ref.x, ref.y, ref.psi + shift, ref.v, ref.kappa, ref.s)
        x_bar, u_bar = self._nominal(x0)
        A, B, cc = linearize_horizon(x_bar, u_bar, p, c.dt)
        fric = friction_rows(x_bar, mu, c)
        H = obstacle_halfplanes(x_bar, obstacles, p.length, p.width, c, c.dt)
        Pm, q, Acon, l, u = qp.build(A, B, cc, x0, ref, self._u_applied, fric, H, brake_request)
        z0 = np.zeros(qp.n_var)
        z0[: qp.iu0] = x_bar.ravel()
        z0[qp.iu0 : qp.sf0] = u_bar.ravel()
        z = self._solve_qp(Pm, q, Acon, l, u, z0)
        solve_time = time.perf_counter() - t0
        if z is not None:
            self._fails = 0
            status = self._status
            self._plan_x = z[: qp.iu0].reshape(N + 1, 6)
            self._plan_u = z[qp.iu0 : qp.sf0].reshape(N, 2)
            sf, so = z[qp.sf0 : qp.so0], z[qp.so0 :]
            sf_max, so_max, cost = float(sf.max()), float(so.max()), self._cost
        else:
            self._fails += (
                1  # a failure before any plan exists goes straight to fallback_brake (nothing to shift)
            )
            sf_max = so_max = cost = 0.0
            if self._fails <= 2 and self._plan_x is not None:
                status = "fallback_shift"
                self._plan_x, self._plan_u = x_bar, u_bar
            else:
                status = "fallback_brake"
                u_hold = np.array([self._u_applied[0], c.ax_min])
                self._plan_u = np.tile(u_hold, (N, 1))
                self._plan_x = x_bar.copy()
                for k in range(N):
                    self._plan_x[k + 1] = rk4_step(self._plan_x[k], u_hold, p, c.dt)
        control = Control(delta=float(self._plan_u[0, 0]), ax=float(self._plan_u[0, 1]))
        self._u_applied = self._plan_u[0].copy()
        return MPCResult(control, status, solve_time, self._plan_x.copy(), sf_max, so_max, cost)
