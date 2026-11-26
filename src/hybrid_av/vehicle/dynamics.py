"""Continuous bicycle models; equations in Plan/01-theory/vehicle-model.md."""

import numpy as np

from hybrid_av.vehicle.params import VehicleParams

V_KIN_SWITCH: float = 5.0  # m/s; below → kinematic model
V_EPS: float = 1.0  # m/s floor used inside tyre slip-angle denominators
TAU_KIN: float = 0.1  # s, relaxation time of vy and r in the kinematic model
V_BLEND_LO = 3.0


def _pose_rates(x):
    vx, vy, psi = x[0], x[1], x[5]
    c, s = np.cos(psi), np.sin(psi)
    return vx * c - vy * s, vx * s + vy * c


def f_dynamic(x: np.ndarray, u: np.ndarray, p: VehicleParams) -> np.ndarray:
    vx, vy, r = x[0], x[1], x[2]
    delta, ax = u
    v = max(vx, V_EPS)
    fyf = p.Cf * (delta - (vy + p.a * r) / v)
    fyr = p.Cr * (-(vy - p.b * r) / v)
    xd, yd = _pose_rates(x)
    cd = np.cos(delta)
    return np.array(
        [
            ax + r * vy,
            (fyf * cd + fyr) / p.m - r * vx,
            (p.a * fyf * cd - p.b * fyr) / p.Iz,
            xd,
            yd,
            r,
        ],
        dtype=np.float64,
    )


def f_kinematic(x: np.ndarray, u: np.ndarray, p: VehicleParams) -> np.ndarray:
    vx, vy, r = x[0], x[1], x[2]
    delta, ax = u
    r_kin = vx * np.tan(delta) / p.L
    xd, yd = _pose_rates(x)
    return np.array(
        [ax, (p.b * r_kin - vy) / TAU_KIN, (r_kin - r) / TAU_KIN, xd, yd, r],
        dtype=np.float64,
    )


def f_blended(x: np.ndarray, u: np.ndarray, p: VehicleParams) -> np.ndarray:
    w = float(np.clip((x[0] - V_BLEND_LO) / (V_KIN_SWITCH - V_BLEND_LO), 0.0, 1.0))
    return (1.0 - w) * f_kinematic(x, u, p) + w * f_dynamic(x, u, p)


def jacobians(x: np.ndarray, u: np.ndarray, p: VehicleParams) -> tuple[np.ndarray, np.ndarray]:
    """Exact (A, B) = (df/dx, df/du) of f_blended, incl. dw/dvx and the V_EPS clamp."""
    vx, vy, r, psi = x[0], x[1], x[2], x[5]
    delta = u[0]
    span = V_KIN_SWITCH - V_BLEND_LO
    w = float(np.clip((vx - V_BLEND_LO) / span, 0.0, 1.0))
    dw = 1.0 / span if V_BLEND_LO < vx < V_KIN_SWITCH else 0.0

    # dynamic model; the clamp max(vx, V_EPS) has zero slope w.r.t. vx below V_EPS
    v = max(vx, V_EPS)
    dv = 1.0 if vx > V_EPS else 0.0
    cd, sd = np.cos(delta), np.sin(delta)
    sf, sr = (vy + p.a * r) / v, (vy - p.b * r) / v
    fyf = p.Cf * (delta - sf)
    # axle force derivatives: columns [vx, vy, r, delta]
    dfyf = np.array([p.Cf * sf / v * dv, -p.Cf / v, -p.Cf * p.a / v, p.Cf])
    dfyr = np.array([p.Cr * sr / v * dv, -p.Cr / v, p.Cr * p.b / v, 0.0])
    Ad = np.zeros((6, 6))
    Bd = np.zeros((6, 2))
    Ad[0, 1], Ad[0, 2], Bd[0, 1] = r, vy, 1.0
    row1 = (dfyf * cd + dfyr) / p.m
    row2 = (p.a * dfyf * cd - p.b * dfyr) / p.Iz
    row1[3] -= fyf * sd / p.m
    row2[3] -= p.a * fyf * sd / p.Iz
    row1[0] -= r
    row1[2] -= vx
    Ad[1, [0, 1, 2]], Bd[1, 0] = row1[:3], row1[3]
    Ad[2, [0, 1, 2]], Bd[2, 0] = row2[:3], row2[3]

    # kinematic model
    Ak = np.zeros((6, 6))
    Bk = np.zeros((6, 2))
    tn = np.tan(delta)
    dtn = 1.0 / cd**2
    Bk[0, 1] = 1.0
    Ak[1, 0], Ak[1, 1], Bk[1, 0] = (
        p.b * tn / (p.L * TAU_KIN),
        -1.0 / TAU_KIN,
        p.b * vx * dtn / (p.L * TAU_KIN),
    )
    Ak[2, 0], Ak[2, 2], Bk[2, 0] = tn / (p.L * TAU_KIN), -1.0 / TAU_KIN, vx * dtn / (p.L * TAU_KIN)

    A = (1.0 - w) * Ak + w * Ad
    B = (1.0 - w) * Bk + w * Bd
    # pose kinematics, identical in both models
    c, s = np.cos(psi), np.sin(psi)
    A[3, 0], A[3, 1], A[3, 5] = c, -s, -vx * s - vy * c
    A[4, 0], A[4, 1], A[4, 5] = s, c, vx * c - vy * s
    A[5, 2] = 1.0
    A[:, 0] += dw * (f_dynamic(x, u, p) - f_kinematic(x, u, p))
    return A, B


def rk4_step(x: np.ndarray, u: np.ndarray, p: VehicleParams, dt: float, substeps: int = 5) -> np.ndarray:
    """Classic RK4 over `substeps` equal sub-intervals; u held constant over dt."""
    h = dt / substeps
    x = np.asarray(x, dtype=np.float64)
    for _ in range(substeps):
        k1 = f_blended(x, u, p)
        k2 = f_blended(x + 0.5 * h * k1, u, p)
        k3 = f_blended(x + 0.5 * h * k2, u, p)
        k4 = f_blended(x + h * k3, u, p)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
