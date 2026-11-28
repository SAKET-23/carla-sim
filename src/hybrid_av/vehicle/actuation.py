"""Control (delta, ax) → CARLA-style actuator command; see Plan/01-theory/vehicle-model.md §8."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hybrid_av.config import load_dataclass
from hybrid_av.types import ActuatorCommand, Control
from hybrid_av.vehicle.params import VehicleParams


@dataclass(frozen=True, slots=True)
class LongitudinalMap:
    throttle_gain: float = 0.35  # throttle per m/s² of required drive accel
    throttle_bias: float = 0.10
    brake_gain: float = 0.20  # brake per m/s² of required decel
    coast_band: float = 0.2  # |ax| below this → no throttle, no brake

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LongitudinalMap":
        return load_dataclass(cls, path)


def control_to_actuator(u: Control, vx: float, p: VehicleParams, lm: LongitudinalMap) -> ActuatorCommand:
    f = np.interp(abs(vx), p.steer_curve_v, p.steer_curve_k)  # CARLA reduces the wheel angle with speed
    steer = float(np.clip(-u.delta / (p.max_steer_wheel * f), -1.0, 1.0))  # CARLA: positive = right
    throttle = brake = 0.0
    if abs(u.ax) >= lm.coast_band:
        required = u.ax + p.rolling + p.drag_coeff * vx**2
        if required > 0:
            throttle = float(np.clip(lm.throttle_bias + lm.throttle_gain * required, 0.0, 1.0))
        else:
            brake = float(np.clip(lm.brake_gain * -required, 0.0, 1.0))
    return ActuatorCommand(steer=steer, throttle=throttle, brake=brake)
