"""Constant-velocity EKF, state [px, py, vx, vy] in the world frame (tracking-fusion.md §1-4)."""

import numpy as np

_MIN_RANGE = 0.5


def cv_predict(x: np.ndarray, P: np.ndarray, dt: float, q_accel: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = q_accel**2
    Q = q * np.array(
        [
            [dt**3 / 3, 0, dt**2 / 2, 0],
            [0, dt**3 / 3, 0, dt**2 / 2],
            [dt**2 / 2, 0, dt, 0],
            [0, dt**2 / 2, 0, dt],
        ]
    )
    return F @ x, F @ P @ F.T + Q


def _update(x, P, y, H, R):
    """Joseph-form update with innovation y (m,), returns (x, P, nis)."""
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    I_KH = np.eye(len(x)) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
    return x + K @ y, (P_new + P_new.T) / 2, float(y @ np.linalg.solve(S, y))


def update_position(x, P, z, R):
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    return _update(x, P, np.asarray(z) - x[:2], H, R)


def _los(x, sensor_xy):
    d = x[:2] - sensor_xy
    rho = float(np.hypot(d[0], d[1]))
    if rho < _MIN_RANGE:
        raise ValueError(f"track {rho:.2f} m from the radar; a return inside the vehicle is an upstream bug")
    return d, rho


def radar_h(x: np.ndarray, sensor_xy: np.ndarray) -> float:
    d, rho = _los(x, sensor_xy)
    return float(d @ x[2:] / rho)


def radar_jacobian(x: np.ndarray, sensor_xy: np.ndarray) -> np.ndarray:
    d, rho = _los(x, sensor_xy)
    dv = d @ x[2:]
    return np.array([[*(x[2:] / rho - d * dv / rho**3), *(d / rho)]])


def update_radar(x, P, rr, sensor_xy, r_var):
    y = np.array([rr - radar_h(x, sensor_xy)])
    return _update(x, P, y, radar_jacobian(x, sensor_xy), np.array([[r_var]]))
