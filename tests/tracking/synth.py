"""Synthetic constant-velocity targets and noisy box / radar observations in the library convention."""

import math
from dataclasses import dataclass

import numpy as np

from hybrid_av.geometry import rotate, world_to_base
from hybrid_av.perception.base import Box3D, RadarDetection
from hybrid_av.tracking.radar import RadarExtrinsic, sensor_velocity_base
from hybrid_av.types import EgoState, ObjectClass

# half horizontal FOV per radar (simulation-backends.md rig: 30 deg front/rear, 60 deg sides)
_HALF_FOV = {
    "front": math.radians(15),
    "rear": math.radians(15),
    "left": math.radians(30),
    "right": math.radians(30),
}


@dataclass(frozen=True)
class Target:
    x0: float
    y0: float
    vx: float
    vy: float

    def pos(self, t: float) -> np.ndarray:
        return np.array([self.x0 + self.vx * t, self.y0 + self.vy * t])


def ego_at(t, vx=0.0, x0=0.0, y0=0.0, psi=0.0) -> EgoState:
    s = vx * t
    return EgoState(t=t, x=x0 + s * math.cos(psi), y=y0 + s * math.sin(psi), psi=psi, vx=vx, vy=0.0, r=0.0)


def make_target(x0, y0, vx, vy) -> Target:
    return Target(x0, y0, vx, vy)


def box_obs(target: Target, t: float, ego: EgoState, noise_sigma: float, rng) -> Box3D:
    p = world_to_base(target.pos(t), ego) + rng.normal(scale=noise_sigma, size=2)
    heading = math.atan2(target.vy, target.vx) if (target.vx or target.vy) else 0.0
    return Box3D(p[0], p[1], 0.8, 4.5, 1.8, 1.6, heading - ego.psi, ObjectClass.CAR, 1.0)


def radar_obs(target: Target, t: float, ego: EgoState, ext: RadarExtrinsic, rr_sigma: float, rng):
    p = world_to_base(target.pos(t), ego)
    d = p - np.array([ext.x, ext.y])
    az = math.atan2(d[1], d[0]) - ext.yaw
    az = math.atan2(math.sin(az), math.cos(az))
    if abs(az) > _HALF_FOV[ext.sensor_id]:
        return None
    depth = float(np.hypot(*d))
    u = d / depth
    v_obj = rotate(np.array([target.vx, target.vy]), -ego.psi)
    rel = float((v_obj - sensor_velocity_base(ext, ego)) @ u) + rng.normal(scale=rr_sigma)
    return RadarDetection(depth, az, 0.0, rel, ext.sensor_id)
