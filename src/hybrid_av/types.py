"""Shared types; the contract is Plan/02-system-design/interfaces.md section 1."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np


class TrafficLightState(IntEnum):
    NONE = 0  # no light controlling the ego lane within 60 m
    GREEN = 1
    YELLOW = 2
    RED = 3


class ObjectClass(IntEnum):
    UNKNOWN = 0
    CAR = 1
    TRUCK = 2
    PEDESTRIAN = 3
    CYCLIST = 4


@dataclass(frozen=True, slots=True)
class EgoState:
    t: float  # sim time, s
    x: float  # world X, m
    y: float  # world Y, m
    psi: float  # heading, rad, unwrapped (continuous across laps)
    vx: float  # body longitudinal velocity, m/s
    vy: float  # body lateral velocity, m/s (left positive)
    r: float  # yaw rate, rad/s (CCW positive)
    ax: float = 0.0  # measured longitudinal acceleration, m/s²

    def as_mpc_vector(self) -> np.ndarray:
        """[vx, vy, r, x, y, psi], shape (6,)."""
        return np.array([self.vx, self.vy, self.r, self.x, self.y, self.psi], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ObjectState:
    id: int
    cls: ObjectClass
    x: float  # world frame centre, m
    y: float
    z: float
    yaw: float  # world frame, rad
    length: float
    width: float
    height: float
    vx: float  # world frame velocity, m/s
    vy: float
    ax: float = 0.0  # world frame acceleration, m/s² (0 if unknown)
    ay: float = 0.0
    pos_cov: np.ndarray | None = None  # 2x2 world-frame position covariance (None = ground truth)


@dataclass(frozen=True, slots=True)
class TrafficContext:
    light_state: TrafficLightState
    dist_to_stop_line: float  # m along route to the stop line governing the light; inf if none
    dist_to_stop_sign: float  # m along route; inf if none within 100 m
    dist_to_junction: float  # m along route to entry of next junction; inf if none within 100 m
    speed_limit: float  # m/s
    lane_width: float  # m
    road_grade: float  # rad, positive uphill
    surface_roughness: float  # 0..1 surface-condition index (state-vector.md slot 7)


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything the stack needs for one control step. Produced by any Sim backend or by the ROS graph."""

    ego: EgoState
    objects: tuple[ObjectState, ...]
    context: TrafficContext
    friction_true: float  # ground-truth μ under the ego (logging/eval only; never a policy input)
    source: Literal["ground_truth", "perception"]


@dataclass(frozen=True, slots=True)
class Control:
    """MPC output in physical units."""

    delta: float  # front wheel steer, rad
    ax: float  # longitudinal acceleration command, m/s²


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    """What a simulator applies. Same semantics as carla.VehicleControl."""

    steer: float  # [-1, 1], positive = right turn in CARLA convention (converted in actuation.py)
    throttle: float  # [0, 1]
    brake: float  # [0, 1]


@dataclass(frozen=True, slots=True)
class PolicyAction:
    """Policy output after scaling. See policy-mlp.md for the tanh -> physical map."""

    alpha_v: float  # [0.0, 1.2] speed-limit multiplier
    delta_y: float  # [-3.5, 3.5] m lateral offset from route centreline, left positive
    brake: float  # [0, 1]; >= 0.5 means emergency stop request
