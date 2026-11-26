import numpy as np
from scipy.linalg import expm

from hybrid_av.vehicle.dynamics import jacobians, rk4_step
from hybrid_av.vehicle.params import VehicleParams


def linearize(
    x_bar: np.ndarray, u_bar: np.ndarray, p: VehicleParams, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ZOH discretisation (augmented expm); cd makes the model exact under rk4_step at (x_bar, u_bar)."""
    Ac, Bc = jacobians(x_bar, u_bar, p)
    M = np.zeros((8, 8))
    M[:6, :6], M[:6, 6:] = Ac, Bc
    E = expm(M * dt)
    Ad, Bd = E[:6, :6], E[:6, 6:]
    return Ad, Bd, rk4_step(x_bar, u_bar, p, dt) - Ad @ x_bar - Bd @ u_bar


def linearize_horizon(
    x_traj: np.ndarray, u_traj: np.ndarray, p: VehicleParams, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A, B, c = zip(*(linearize(x, u, p, dt) for x, u in zip(x_traj[:-1], u_traj)))
    return np.array(A), np.array(B), np.array(c)
