"""rig_subset.py RIG_YAML PROFILE OUT_YAML: write the sensor subset a G2 profile uses (none/A_norender are not rigs)."""

import sys
from pathlib import Path

import yaml

KEEP = {
    "cameras": {"camera_front", "camera_front_semantic", "imu"},
    "cameras_lidar": {"camera_front", "camera_front_semantic", "imu", "lidar"},
    "cam_rgb": {"camera_front", "imu"},
    "cam_540": {"camera_front", "camera_front_semantic", "imu"},
    "cam_360": {"camera_front", "camera_front_semantic", "imu"},
}
RES = {"cam_540": ("960", "540"), "cam_360": ("640", "360")}
LIGHT = {"channels": "32", "points_per_second": "600000"}


def subset(rig: dict, profile: str) -> dict:
    sensors = [s for s in rig["sensors"] if profile not in KEEP or s["id"] in KEEP[profile]]
    if profile == "full_lidar_light":
        for s in sensors:
            if s["id"] == "lidar":
                s["attributes"].update(LIGHT)
    for s in sensors:
        if profile in RES and s["id"].startswith("camera"):
            s["attributes"]["image_size_x"], s["attributes"]["image_size_y"] = RES[profile]
    return {"sensors": sensors}


if __name__ == "__main__":
    src, profile, out = sys.argv[1:4]
    Path(out).write_text(yaml.safe_dump(subset(yaml.safe_load(Path(src).read_text()), profile)))
