"""Sensor rig from configs/sensors/carla_rig.yaml: radar extrinsics and static transforms. Pure: no rclpy."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from hybrid_av.tracking.radar import RadarExtrinsic

_OPTICAL = (
    0.0,
    0.0,
    0.0,
    -math.pi / 2,
    0.0,
    -math.pi / 2,
)  # camera_front (x fwd) -> optical (z fwd, x right, y down)


@dataclass(frozen=True)
class Rig:
    radars: list[RadarExtrinsic]
    static_transforms: list[tuple[str, str, tuple[float, ...]]]  # (parent, child, x y z roll pitch yaw)


def load_rig(path: str | Path) -> Rig:
    radars, transforms, seen = [], [("base_link", "sensor_kit_link", (0.0,) * 6)], set()
    for s in yaml.safe_load(Path(path).read_text())["sensors"]:
        m = s["mount"]
        pose = (m["x"], m["y"], m["z"], m.get("roll", 0.0), m.get("pitch", 0.0), m["yaw"])
        if s["frame"] not in seen:  # the RGB and semantic cameras share one frame
            seen.add(s["frame"])
            transforms.append(("sensor_kit_link", s["frame"], tuple(float(v) for v in pose)))
        if s["id"].startswith("radar_"):
            radars.append(RadarExtrinsic(s["id"].removeprefix("radar_"), m["x"], m["y"], m["z"], m["yaw"]))
    transforms.append(("camera_front", "camera_front_optical", _OPTICAL))
    return Rig(radars, transforms)


def lidar_to_base(points: np.ndarray, mount: tuple[float, ...]) -> np.ndarray:
    """(N,4) x,y,z,intensity from the lidar frame to base_link. shortcut: yaw-only mounts, add roll/pitch if the rig tilts."""
    x, y, z, roll, pitch, yaw = mount
    if roll or pitch:
        raise ValueError("lidar_to_base supports yaw-only mounts")
    c, s = math.cos(yaw), math.sin(yaw)
    out = points.copy()
    out[:, 0] = c * points[:, 0] - s * points[:, 1] + x
    out[:, 1] = s * points[:, 0] + c * points[:, 1] + y
    out[:, 2] = points[:, 2] + z
    return out
