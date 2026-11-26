"""Test-only car version of Misin's LPV-MPC (Plan/00-raw/research/misin-lpv-mpc.md) on the path-frame lateral-error
model xi = [vx, vy, r, e_lat, e_psi]: A, B, E frozen at the current state, ZOH, state augmented with u_{k-1},
known disturbance r_ref = v_ref * kappa, condensed unconstrained QP in delta-u, clipped to the hard limits afterwards."""

import time

import numpy as np
from scipy.linalg import expm

from hybrid_av.geometry import unwrap_near
from hybrid_av.mpc.controller import MPCConfig, MPCResult
from hybrid_av.types import Control
from hybrid_av.vehicle.dynamics import rk4_step
from hybrid_av.vehicle.params import VehicleParams


def lpv_matrices(xi, p: VehicleParams, delta: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xi_dot = A xi + B u + E r_ref, exact at the scheduling point xi. The cos(delta) force projection is
    scheduled on `delta` (the applied steering)."""
    vx, e_psi = xi[0], xi[4]
    v = max(vx, 1.0)
    c = np.cos(delta)
    A, B, E = np.zeros((5, 5)), np.zeros((5, 2)), np.zeros(5)
    A[0, 1] = xi[2]  # r * vy
    A[1, 1] = -(p.Cf * c + p.Cr) / (p.m * v)
    A[1, 2] = -(p.a * p.Cf * c - p.b * p.Cr) / (p.m * v) - vx
    A[2, 1] = (-p.a * p.Cf * c + p.b * p.Cr) / (p.Iz * v)
    A[2, 2] = -(p.a**2 * p.Cf * c + p.b**2 * p.Cr) / (p.Iz * v)
    A[3, 1] = np.cos(e_psi)
    A[3, 4] = vx * np.sinc(e_psi / np.pi)  # vx sin(e_psi) = A[3,4] e_psi
    A[4, 2] = 1.0
    B[0, 1] = 1.0
    B[1, 0] = p.Cf * c / p.m
    B[2, 0] = p.a * p.Cf * c / p.Iz
    E[4] = -1.0
    return A, B, E


class MisinLPVMPC:
    def __init__(self, params: VehicleParams, cfg: MPCConfig) -> None:
        self.params, self.cfg = params, cfg
        self.reset()

    def reset(self) -> None:
        self._u_applied = np.zeros(2)

    def solve(self, ego, ref, mu, obstacles=(), brake_request=False) -> MPCResult:
        t0 = time.perf_counter()
        c, p, N = self.cfg, self.params, self.cfg.N
        u_prev = self._u_applied
        x0 = rk4_step(ego.as_mpc_vector(), u_prev, p, c.dt)  # same delay compensation as BicycleMPC
        psi_r = ref.psi[0]
        dX, dY = x0[3] - ref.x[0], x0[4] - ref.y[0]
        e_lat = -np.sin(psi_r) * dX + np.cos(psi_r) * dY
        e_psi = x0[5] - unwrap_near(psi_r, x0[5])
        xi0 = np.array([x0[0], x0[1], x0[2], e_lat, e_psi])
        A, B, E = lpv_matrices(xi0, p, u_prev[0])
        M = expm(np.block([[A, B, E[:, None]], [np.zeros((3, 8))]]) * c.dt)
        Ad, Bd, Ed = M[:5, :5], M[:5, 5:7], M[:5, 7]
        At = np.block([[Ad, Bd], [np.zeros((2, 5)), np.eye(2)]])
        Bt = np.vstack([Bd, np.eye(2)])
        Et = np.concatenate([Ed, np.zeros(2)])
        Ct = np.zeros((3, 7))
        Ct[0, 0], Ct[1, 3], Ct[2, 4] = 1.0, 1.0, 1.0
        # condensed prediction x~_{i+1} = Ahat_i x~_0 + sum_j Cbar_ij du_j + sum_j pw[i-j] Et d_j
        pw = [np.eye(7)]
        for _ in range(N):
            pw.append(At @ pw[-1])
        Ahat = np.vstack(pw[1:])
        d = ref.v[:N] * ref.kappa[:N]
        Cbar = np.zeros((7 * N, 2 * N))
        dist = np.zeros(7 * N)
        for i in range(N):
            for j in range(i + 1):
                Cbar[7 * i : 7 * i + 7, 2 * j : 2 * j + 2] = pw[i - j] @ Bt
                dist[7 * i : 7 * i + 7] += pw[i - j] @ Et * d[j]
        Qy = np.diag([c.Q[0], c.w_lat, c.w_psi])
        Qb = np.zeros((3 * N, 3 * N))
        for i in range(N):
            Qb[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = Qy * (c.P_scale if i == N - 1 else 1.0)
        Cy = np.kron(np.eye(N), Ct)
        x_t0 = np.concatenate([xi0, u_prev])
        r = np.column_stack([ref.v[1:], np.zeros((N, 2))]).ravel()
        CyC = Cy @ Cbar
        H = CyC.T @ Qb @ CyC + np.kron(np.eye(N), np.diag(c.R_delta))
        f = CyC.T @ Qb @ (Cy @ (Ahat @ x_t0 + dist) - r)
        du = np.linalg.solve(H, -f)
        d_delta = np.clip(du[0], -c.ddelta_max, c.ddelta_max)
        delta = float(np.clip(u_prev[0] + d_delta, -c.delta_max, c.delta_max))
        ax = float(np.clip(u_prev[1] + du[1], c.ax_min, c.ax_max))
        # The path-frame model has no world pose: `predicted` is the (N+1, 6) current state repeated (placeholder).
        pred = np.tile(x0, (N + 1, 1))
        self._u_applied = np.array([delta, ax])
        cost = float(0.5 * du @ H @ du + f @ du)
        return MPCResult(
            Control(delta=delta, ax=ax), "solved", time.perf_counter() - t0, pred, 0.0, 0.0, cost
        )
