import math
from pathlib import Path

import numpy as np
import pytest
from hybrid_av_nodes.rig import lidar_to_base, load_rig

RIG = Path(__file__).parents[4] / "configs/sensors/carla_rig.yaml"


def test_rig_has_four_radars_and_sensor_frames():
    r = load_rig(RIG)
    assert {x.sensor_id for x in r.radars} == {"front", "rear", "left", "right"}
    rear = next(x for x in r.radars if x.sensor_id == "rear")
    assert rear.x == pytest.approx(-2.4) and abs(abs(rear.yaw) - math.pi) < 1e-9
    children = {c for _, c, _ in r.static_transforms}
    assert {"sensor_kit_link", "lidar_top", "camera_front", "camera_front_optical", "imu",
            "radar_front", "radar_rear", "radar_left", "radar_right"} <= children  # fmt: skip


def test_optical_frame_and_lidar_mount():
    r = load_rig(RIG)
    tf = {c: (p, xyzrpy) for p, c, xyzrpy in r.static_transforms}
    assert tf["camera_front_optical"][0] == "camera_front"
    assert tf["camera_front_optical"][1] == pytest.approx((0, 0, 0, -math.pi / 2, 0, -math.pi / 2))
    assert tf["lidar_top"] == ("sensor_kit_link", (0.0, 0.0, 1.9, 0.0, 0.0, 0.0))
    assert tf["sensor_kit_link"][0] == "base_link"


def test_lidar_to_base_applies_mount():
    out = lidar_to_base(np.array([[1.0, 0.0, 0.0, 0.5]], np.float32), (0.5, 0.0, 1.9, 0.0, 0.0, math.pi / 2))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.5, 1.0, 1.9, 0.5]], atol=1e-6)
