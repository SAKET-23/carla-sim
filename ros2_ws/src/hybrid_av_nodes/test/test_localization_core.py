import math

import pytest
from hybrid_av_nodes.localization_core import LocalizationCore

from hybrid_av.types import EgoState


def test_alignment_from_first_ground_truth():
    c = LocalizationCore()
    c.on_ground_truth(0.0, EgoState(t=0, x=100.0, y=50.0, psi=math.pi / 2, vx=0, vy=0, r=0))
    assert c.on_odom(0.0, 0.0, 0.0, 0.0).x == pytest.approx(100.0)  # odom origin = start pose
    c.on_wheel(0.05, 10.0, 0.1)
    c.on_imu(0.05, 0.02)
    e = c.on_odom(0.05, 1.0, 0.0, 0.0)  # 1 m forward in odom
    assert (e.x, e.y, e.psi) == pytest.approx((100.0, 51.0, math.pi / 2))
    assert (e.vx, e.vy, e.r) == pytest.approx((10.0, 0.1, 0.02))


def test_divergence_flag_never_changes_output():
    c = LocalizationCore(divergence_m=2.0)
    c.on_ground_truth(0.0, EgoState(t=0, x=0, y=0, psi=0, vx=0, vy=0, r=0))
    c.on_odom(0.0, 0, 0, 0)
    c.on_ground_truth(1.0, EgoState(t=1, x=10, y=0, psi=0, vx=0, vy=0, r=0))
    e = c.on_odom(1.0, 13.0, 0.0, 0.0)
    assert c.localization_failed and e.x == pytest.approx(13.0)


def test_no_output_before_alignment():
    assert LocalizationCore().on_odom(0.0, 1.0, 2.0, 0.0) is None


def test_map_to_odom_maps_odom_pose_to_map_pose():
    c = LocalizationCore()
    c.on_ground_truth(0.0, EgoState(t=0, x=100.0, y=50.0, psi=math.pi / 2, vx=0, vy=0, r=0))
    c.on_odom(0.0, 2.0, 1.0, 0.1)  # odom does not start at the origin
    tx, ty, tyaw = c.map_to_odom()
    e = c.on_odom(0.05, 3.0, 1.0, 0.1)
    assert (
        tx + math.cos(tyaw) * 3.0 - math.sin(tyaw) * 1.0,
        ty + math.sin(tyaw) * 3.0 + math.cos(tyaw) * 1.0,
    ) == (pytest.approx((e.x, e.y)))
    assert tyaw + 0.1 == pytest.approx(e.psi)
