"""Control (delta, ax) → CARLA-style actuator command; see Plan/01-theory/vehicle-model.md §8."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hybrid_av.config import load_dataclass
from hybrid_av.types import ActuatorCommand, Control
from hybrid_av.vehicle.params import VehicleParams

HOLD_SPEED, HOLD_BRAKE = 0.3, 0.3  # table path: below this speed a non-positive request holds the car


@dataclass(frozen=True, slots=True)
class LongitudinalMap:
    throttle_gain: float = 0.35  # throttle per m/s² of required drive accel
    throttle_bias: float = 0.10
    brake_gain: float = 0.20  # brake per m/s² of required decel
    coast_band: float = 0.2  # |ax| below this → no throttle, no brake
    # Optional table (replaces the linear law, rolling/drag_coeff are then ignored): table_accels[i][j] is the NET
    # accel at table_speeds[i] m/s and table_throttles[j] (non-decreasing in j; throttle 0 is the coast row, <= 0).
    # With a table, brake supplies the decel beyond coasting and coast_band is the deadband below the coast accel.
    table_speeds: tuple[float, ...] = ()
    table_throttles: tuple[float, ...] = ()
    table_accels: tuple[tuple[float, ...], ...] = ()
    # Optional brake table (table maps only): brake_decels[i] is the extra decel beyond coasting at brake_levels[i]
    # (levels start at 0.0; decels >= 0, non-decreasing). Empty -> the linear brake_gain law.
    brake_levels: tuple[float, ...] = ()
    brake_decels: tuple[float, ...] = ()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LongitudinalMap":
        return load_dataclass(cls, path)


def _net_accels(lm: LongitudinalMap, vx: float) -> np.ndarray:
    """Net accel per table throttle level at `vx` (interpolated across the speed nodes)."""
    net = np.array([np.interp(abs(vx), lm.table_speeds, col) for col in zip(*lm.table_accels, strict=True)])
    if abs(vx) < lm.table_speeds[0]:  # coast decel is ~proportional to speed: goes to 0 at standstill
        net[0] = lm.table_accels[0][0] * abs(vx) / lm.table_speeds[0]
    return net


def coast_accel(lm: LongitudinalMap, vx: float) -> float:
    """Net accel at throttle 0 and no brake at `vx` (table maps only)."""
    return float(_net_accels(lm, vx)[0])


def drive_throttle(lm: LongitudinalMap, vx: float, required: float) -> float:
    """Throttle in [0, 1] for `required` m/s²: the table inverted at `vx` (required = desired net accel), else the
    linear law (required = drive accel beyond the resistances)."""
    if lm.table_speeds:
        return float(np.clip(np.interp(required, _net_accels(lm, vx), lm.table_throttles), 0.0, 1.0))
    return float(np.clip(lm.throttle_bias + lm.throttle_gain * required, 0.0, 1.0))


def brake_for(lm: LongitudinalMap, d: float) -> float:
    """Smallest brake in [0, 1] giving `d` > 0 m/s² of decel beyond coasting. The flat dead zone is skipped (a small d
    gives its upper edge) and a d past the table gives its first saturating level."""
    if not lm.brake_levels:
        return float(np.clip(lm.brake_gain * d, 0.0, 1.0))
    lv, dec = np.array(lm.brake_levels), np.array(lm.brake_decels)
    keep = np.flatnonzero(np.diff(dec, prepend=-1.0) > 0)  # first level of each strict rise
    keep[0] = np.flatnonzero(dec == dec[0])[-1]  # the dead zone starts at its upper edge
    return float(np.clip(np.interp(d, dec[keep], lv[keep]), 0.0, 1.0))


def control_to_actuator(u: Control, vx: float, p: VehicleParams, lm: LongitudinalMap) -> ActuatorCommand:
    f = np.interp(abs(vx), p.steer_curve_v, p.steer_curve_k)  # CARLA reduces the wheel angle with speed
    steer = float(np.clip(-u.delta / (p.max_steer_wheel * f), -1.0, 1.0))  # CARLA: positive = right
    throttle = brake = 0.0
    if lm.table_speeds:
        a0 = coast_accel(lm, vx)
        if abs(vx) < HOLD_SPEED and u.ax <= 0:  # standstill hold
            brake = HOLD_BRAKE
        elif u.ax >= a0:
            throttle = drive_throttle(lm, vx, u.ax)
        elif u.ax >= a0 - lm.coast_band:
            pass  # coast
        else:
            brake = brake_for(lm, a0 - u.ax)
    elif abs(u.ax) >= lm.coast_band:
        required = u.ax + p.rolling + p.drag_coeff * vx**2
        if required > 0:
            throttle = drive_throttle(lm, vx, required)
        else:
            brake = float(np.clip(lm.brake_gain * -required, 0.0, 1.0))
    return ActuatorCommand(steer=steer, throttle=throttle, brake=brake)
