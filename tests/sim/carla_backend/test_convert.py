import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hybrid_av.sim.carla_backend import convert as C


def test_position_and_yaw():
    assert C.loc_to_lib(10.0, 3.0, 0.5) == (10.0, -3.0, 0.5)
    assert C.yaw_to_lib(90.0) == pytest.approx(-math.pi / 2)  # CARLA +90° = clockwise = library right
    assert C.lib_to_yaw(C.yaw_to_lib(37.0)) == pytest.approx(37.0)


def test_left_turn_round_trip():
    # Library: heading 0 → turning left means psi increases, y increases, r > 0.
    # CARLA reports this as yaw decreasing (deg), y decreasing, angular_velocity.z < 0 (deg/s).
    e = C.ego_from_actor(
        transform=NS(location=NS(x=5.0, y=-1.0, z=0.0), rotation=NS(yaw=-10.0, pitch=0, roll=0)),
        velocity=NS(x=9.85, y=-1.74, z=0.0),
        angular_velocity=NS(x=0, y=0, z=-20.0),
        acceleration=NS(x=0, y=0, z=0),
        t=1.0,
        unwrap=lambda p: p,
    )
    assert e.y == pytest.approx(1.0) and e.psi == pytest.approx(math.radians(10.0))
    assert e.r == pytest.approx(math.radians(20.0))
    assert e.vx == pytest.approx(10.0, abs=0.01) and abs(e.vy) < 0.01  # velocity along heading


def test_lidar_flip_is_one_view():
    raw = np.array([1, 2, 3, 0.5, 4, -5, 6, 0.25], np.float32)
    np.testing.assert_array_equal(C.lidar_to_lib(raw), [[1, -2, 3, 0.5], [4, 5, 6, 0.25]])
    assert raw[1] == 2  # input buffer untouched


def test_radar_layout_and_azimuth():
    raw = np.array([-3.0, 0.2, 0.01, 25.0], np.float32)
    (d,) = C.radar_to_lib(raw, "front")
    assert (d.velocity, d.azimuth, d.altitude, d.depth) == pytest.approx((-3.0, -0.2, 0.01, 25.0))


def test_imu_axes():
    a, g = C.imu_to_lib(NS(x=1.0, y=2.0, z=9.8), NS(x=0.1, y=0.2, z=0.3))
    np.testing.assert_allclose(a, [1.0, -2.0, 9.8])
    np.testing.assert_allclose(g, [-0.1, 0.2, -0.3])


def test_object_and_cyclist():
    o = C.object_from_actor(
        5,
        "vehicle.diamondback.century",
        NS(location=NS(x=1, y=2, z=0), rotation=NS(yaw=0, pitch=0, roll=0)),
        NS(x=0.9, y=0.3, z=0.8),
        NS(x=0, y=0, z=0.8),
        NS(x=0, y=0, z=0),
        number_of_wheels=2,
    )
    assert o.cls.name == "CYCLIST" and o.length == pytest.approx(1.8) and o.y == pytest.approx(-2.0)


def test_two_wheeler_not_in_bike_list_is_cyclist():
    o = C.object_from_actor(
        6, "vehicle.foo.bar", NS(location=NS(x=0, y=0, z=0), rotation=NS(yaw=0)),
        NS(x=1, y=1, z=1), NS(x=0, y=0, z=0), NS(x=0, y=0, z=0), number_of_wheels=2,
    )  # fmt: skip
    assert o.cls.name == "CYCLIST"


def test_bbox_centre_offset_follows_yaw():
    # actor yawed CARLA +90° (library -90°); bbox centre 1 m ahead in the actor frame → library -y
    o = C.object_from_actor(
        1, "vehicle.audi.a2", NS(location=NS(x=0, y=0, z=0), rotation=NS(yaw=90.0)),
        NS(x=2, y=1, z=1), NS(x=1, y=0, z=0), NS(x=0, y=0, z=0),
    )  # fmt: skip
    assert (o.x, o.y) == pytest.approx((0.0, -1.0), abs=1e-9)


def test_bbox_into_base_link():
    from hybrid_av.types import EgoState, ObjectClass

    ego = EgoState(t=0, x=0, y=0, psi=math.pi / 2, vx=0, vy=0, r=0)  # facing library +y
    b = C.bbox_to_box3d(
        NS(location=NS(x=0, y=-10, z=0), rotation=NS(yaw=-90.0, pitch=0, roll=0)),
        NS(x=2.0, y=1.0, z=0.75),
        NS(x=0, y=0, z=0.75),
        ego,
        cls=ObjectClass.CAR,
    )
    assert (b.x, b.y, b.yaw) == pytest.approx((10.0, 0.0, 0.0))  # 10 m straight ahead, same heading


def test_mount_and_steer():
    (loc, yaw) = C.mount_to_carla(0.0, 1.0, 0.7, math.pi / 2)  # left radar
    assert loc == (0.0, -1.0, 0.7) and yaw == pytest.approx(-90.0)


def test_steer_curve_from_physics_converts_kmh_to_ms():
    class V:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class Phys:
        steering_curve = (V(0, 1.0), V(36, 0.9))

    assert C.steer_curve_from_physics(Phys()) == ((0.0, 10.0), (1.0, 0.9))
    assert C.steer_curve_from_physics(object()) == ((0.0,), (1.0,))


def test_module_does_not_import_carla():
    import sys

    assert "carla" not in sys.modules
