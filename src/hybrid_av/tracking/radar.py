"""Radar detection geometry and ego-motion compensation (tracking-fusion.md §4)."""

import math
from dataclasses import dataclass

import numpy as np

from hybrid_av.perception.base import RadarDetection
from hybrid_av.types import EgoState


@dataclass(frozen=True, slots=True)
class RadarExtrinsic:
    sensor_id: str
    x: float
    y: float
    z: float
    yaw: float  # in base_link


def detection_to_base(det: RadarDetection, ext: RadarExtrinsic) -> np.ndarray:
    """(3,) detection position in base_link."""
    ca = math.cos(det.altitude)
    a = ext.yaw + det.azimuth
    return np.array(
        [
            ext.x + det.depth * ca * math.cos(a),
            ext.y + det.depth * ca * math.sin(a),
            ext.z + det.depth * math.sin(det.altitude),
        ]
    )


def sensor_velocity_base(ext: RadarExtrinsic, ego: EgoState) -> np.ndarray:
    """(2,) velocity of the sensor in base_link."""
    return np.array([ego.vx - ego.r * ext.y, ego.vy + ego.r * ext.x])


def compensated_range_rate(det: RadarDetection, ext: RadarExtrinsic, ego: EgoState) -> float:
    """Target's own horizontal radial speed, with the ego's motion removed."""
    a = ext.yaw + det.azimuth
    u = np.array([math.cos(a), math.sin(a)])
    return float(det.velocity / math.cos(det.altitude) + sensor_velocity_base(ext, ego) @ u)
