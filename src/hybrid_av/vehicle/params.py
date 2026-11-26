from dataclasses import dataclass
from pathlib import Path

from hybrid_av.config import load_dataclass


@dataclass(frozen=True, slots=True)
class VehicleParams:
    m: float = 1500.0  # kg
    Iz: float = 2250.0  # kg m²
    a: float = 1.2  # CG → front axle, m
    b: float = 1.6  # CG → rear axle, m
    Cf: float = 19000.0  # N/rad, front axle cornering stiffness
    Cr: float = 20000.0  # N/rad, rear axle cornering stiffness
    max_steer: float = 0.70  # rad, MPC steering bound
    max_steer_wheel: float = 0.70  # rad, physical full-lock wheel angle (CARLA steer=±1)
    length: float = 4.7  # m, bounding box
    width: float = 1.9  # m
    max_accel: float = 2.5  # m/s² at full throttle (low speed)
    max_decel: float = 4.5  # m/s² commanded at brake=1 used by MPC bounds
    drag_coeff: float = 0.0004  # 1/m, a_drag = drag_coeff * vx²
    rolling: float = 0.15  # m/s², constant rolling resistance decel

    @property
    def L(self) -> float:
        return self.a + self.b

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VehicleParams":
        return load_dataclass(cls, path)
