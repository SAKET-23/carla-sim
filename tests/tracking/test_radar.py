import math

import numpy as np
import pytest

from hybrid_av.perception.base import RadarDetection
from hybrid_av.tracking.radar import (
    RadarExtrinsic,
    compensated_range_rate,
    detection_to_base,
    sensor_velocity_base,
)
from hybrid_av.types import EgoState

FRONT = RadarExtrinsic("front", x=2.4, y=0.0, z=0.7, yaw=0.0)
LEFT = RadarExtrinsic("left", x=0.0, y=1.0, z=0.7, yaw=math.pi / 2)


def ego(vx=10.0, r=0.0):
    return EgoState(t=0, x=0, y=0, psi=0, vx=vx, vy=0, r=r)


def test_detection_position_front_and_left():
    np.testing.assert_allclose(
        detection_to_base(RadarDetection(20.0, 0.0, 0.0, 0.0, "front"), FRONT), [22.4, 0, 0.7], atol=1e-9
    )
    np.testing.assert_allclose(
        detection_to_base(RadarDetection(5.0, 0.0, 0.0, 0.0, "left"), LEFT), [0, 6.0, 0.7], atol=1e-9
    )


def test_stationary_target_compensates_to_zero():
    det = RadarDetection(depth=20.0, azimuth=0.0, altitude=0.0, velocity=-10.0, sensor_id="front")
    assert compensated_range_rate(det, FRONT, ego(10.0)) == pytest.approx(0.0, abs=1e-9)


def test_yaw_rate_lever_arm():
    np.testing.assert_allclose(sensor_velocity_base(FRONT, ego(10.0, r=0.5)), [10.0, 1.2], atol=1e-9)
    # target straight left of the front radar: u = (0, 1); measured relative = -1.2 for a stationary target
    det = RadarDetection(depth=5.0, azimuth=math.pi / 2, altitude=0.0, velocity=-1.2, sensor_id="front")
    assert compensated_range_rate(det, FRONT, ego(10.0, r=0.5)) == pytest.approx(0.0, abs=1e-9)


def test_compensation_with_yaw_rate_tracking_md_case():
    # tracking.md §3: ego 10 m/s, yaw rate 0.3 rad/s, stationary target → compensated |rr| < 0.05
    e = ego(10.0, r=0.3)
    for az in (-0.4, 0.0, 0.25):
        p = np.array([2.4, 0.0]) + 20.0 * np.array([np.cos(az), np.sin(az)])
        v_rel = -sensor_velocity_base(FRONT, e) @ (p - [2.4, 0.0]) / 20.0
        det = RadarDetection(depth=20.0, azimuth=az, altitude=0.0, velocity=v_rel, sensor_id="front")
        assert abs(compensated_range_rate(det, FRONT, e)) < 0.05


def test_moving_target_keeps_own_radial_speed():
    det = RadarDetection(depth=30.0, azimuth=0.0, altitude=0.0, velocity=-10.0 + 4.0, sensor_id="front")
    assert compensated_range_rate(det, FRONT, ego(10.0)) == pytest.approx(4.0)
