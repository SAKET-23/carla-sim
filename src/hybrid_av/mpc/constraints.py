from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hybrid_av.mpc.controller import MPCConfig
    from hybrid_av.types import ObjectState

G = 9.81


def friction_rows(x_bar: np.ndarray, mu: float, cfg: MPCConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """|coef_k . x_k + offset_k| <= bound_k + s_f,k for k = 1..N, linearising vx*r about x_bar."""
    vx, r = x_bar[1:, 0], x_bar[1:, 2]
    coef = np.zeros((len(vx), 6))
    coef[:, 0], coef[:, 2] = r, vx
    return coef, -vx * r, np.full(len(vx), cfg.friction_margin * mu * G)


def _support(n: np.ndarray, yaw: float, length: float, width: float) -> float:
    t = np.array([np.cos(yaw), np.sin(yaw)])
    return 0.5 * length * abs(n @ t) + 0.5 * width * abs(n @ np.array([-t[1], t[0]]))


def obstacle_halfplanes(
    x_bar: np.ndarray,
    obstacles: Sequence[ObjectState],
    ego_length: float,
    ego_width: float,
    cfg: MPCConfig,
    dt: float,
) -> np.ndarray:
    """Rows [a, b, c] with a*X + b*Y <= c + s_o; unused slots are [0, 0, 1]."""
    N = len(x_bar) - 1
    H = np.tile([0.0, 0.0, 1.0], (N, cfg.max_obstacles, 1))
    near = sorted(obstacles, key=lambda o: np.hypot(o.x - x_bar[0, 3], o.y - x_bar[0, 4]))
    for j, o in enumerate(near[: cfg.max_obstacles]):
        for k in range(1, N + 1):
            p = np.array([o.x + o.vx * k * dt, o.y + o.vy * k * dt])
            d = x_bar[k, 3:5] - p
            dist = np.hypot(*d)
            n = d / dist if dist > 1e-6 else np.array([np.cos(o.yaw), np.sin(o.yaw)])
            clearance = (
                _support(n, o.yaw, o.length, o.width)
                + _support(n, x_bar[k, 5], ego_length, ego_width)
                + cfg.obstacle_margin
            )
            H[k - 1, j] = [-n[0], -n[1], -n @ p - clearance]
    return H
