"""Tyre-utilisation friction estimator; algorithm in Plan/01-theory/vehicle-model.md section 6."""

import numpy as np

from hybrid_av.types import EgoState
from hybrid_av.vehicle.dynamics import f_dynamic
from hybrid_av.vehicle.params import VehicleParams

G = 9.81
MU_LO, MU_HI = 0.1, 1.2
V_MIN = 3.0  # m/s, below this the saturation test is off and the linear model is re-seeded
TAU_RECOVER = 3.0  # s
AY_MIN = 1.0  # m/s^2, model lateral acceleration needed to judge saturation
RHO_SAT = 0.7
N_SAT = 3


class FrictionEstimator:
    def __init__(self, params: VehicleParams, mu_init: float = 1.0, tau: float = 0.5) -> None:
        self.p, self.tau = params, tau
        self.reset(mu_init)

    def reset(self, mu_init: float = 1.0) -> None:
        self._prior = mu_init
        self._mu = float(np.clip(mu_init, MU_LO, MU_HI))
        self._vy_r: np.ndarray | None = None  # parallel linear-tyre model state [vy, r]
        self._count = 0

    def update(self, ego: EgoState, delta: float, dt: float) -> float:
        vx, ay = ego.vx, ego.vx * ego.r
        u = float(np.hypot(ego.ax, ay)) / G
        saturated = False
        if vx < V_MIN or self._vy_r is None:
            self._vy_r = np.array([ego.vy, ego.r])
            self._count = 0
        else:
            h = dt / 5
            for _ in range(5):
                x = np.array([vx, self._vy_r[0], self._vy_r[1], 0.0, 0.0, 0.0])
                d = f_dynamic(x, np.array([delta, 0.0]), self.p)
                self._vy_r = self._vy_r + h * d[1:3]
            ay_model = vx * self._vy_r[1]
            sat = abs(ay_model) > AY_MIN and ay / ay_model < RHO_SAT
            self._count = self._count + 1 if sat else 0
            saturated = self._count >= N_SAT
        if saturated:
            self._mu += dt / self.tau * (u / 0.9 - self._mu)
        else:
            self._mu += dt / TAU_RECOVER * (self._prior - self._mu)
        self._mu = float(np.clip(max(self._mu, u / 0.95), MU_LO, MU_HI))
        return self._mu
