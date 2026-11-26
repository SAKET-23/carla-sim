"""Aligns KISS-ICP odometry to the map once at start. Pure core: no rclpy.

The map -> odom alignment is fixed from the first ground-truth pose and never updated (initialisation only: after
that the output is odometry, so ATE excludes nothing). Later ground-truth poses feed the divergence monitor, which
raises `localization_failed` but never changes the output (01-theory/localization.md section 3).
"""

import math

from hybrid_av.types import EgoState


class LocalizationCore:
    def __init__(self, divergence_m: float = 2.0) -> None:
        self.divergence_m = divergence_m
        self.localization_failed = False
        self._gt: EgoState | None = None  # latest ground truth
        self._gt0: EgoState | None = None
        self._align: tuple[float, float, float] | None = None  # map -> odom (x, y, yaw)
        self._vx = self._vy = self._r = 0.0

    def on_ground_truth(self, t: float, ego: EgoState) -> None:
        self._gt = ego
        self._gt0 = self._gt0 or ego

    def on_wheel(self, t: float, vx: float, vy: float) -> None:
        self._vx, self._vy = vx, vy

    def on_imu(self, t: float, gz: float) -> None:
        self._r = gz

    def map_to_odom(self) -> tuple[float, float, float]:
        return self._align if self._align is not None else (0.0, 0.0, 0.0)

    def on_odom(self, t: float, x: float, y: float, yaw: float) -> EgoState | None:
        if self._align is None:
            if self._gt0 is None:
                return None
            g = self._gt0
            dyaw = g.psi - yaw
            c, s = math.cos(dyaw), math.sin(dyaw)
            self._align = (g.x - (c * x - s * y), g.y - (s * x + c * y), dyaw)
        ax, ay, ayaw = self._align
        c, s = math.cos(ayaw), math.sin(ayaw)
        ego = EgoState(t=t, x=ax + c * x - s * y, y=ay + s * x + c * y, psi=yaw + ayaw,
                       vx=self._vx, vy=self._vy, r=self._r)  # fmt: skip
        if self._gt is not None and math.hypot(ego.x - self._gt.x, ego.y - self._gt.y) > self.divergence_m:
            self.localization_failed = True
        return ego
