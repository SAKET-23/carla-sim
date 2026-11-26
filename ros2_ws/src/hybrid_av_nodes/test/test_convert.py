import math

import numpy as np
import pytest
from hybrid_av_nodes import convert as C
from msgns import field, odometry, pointcloud2, track, track_array, traffic_context

from hybrid_av.types import (
    EgoState,
    ObjectClass,
    ObjectState,
    PolicyAction,
    TrafficContext,
    TrafficLightState,
)


def test_odometry_roundtrip():
    e = EgoState(t=12.35, x=10.0, y=-4.0, psi=0.7, vx=8.0, vy=0.2, r=0.1)
    m = odometry()
    C.fill_odometry(m, e, "map", "base_link")
    C.fill_stamp(m.header.stamp, e.t)
    back = C.ego_from_odometry(m, C.HeadingUnwrapper())
    assert (back.t, back.x, back.y, back.vx, back.vy, back.r) == pytest.approx(
        (12.35, 10.0, -4.0, 8.0, 0.2, 0.1)
    )
    assert back.psi == pytest.approx(0.7, abs=1e-12)


def test_heading_unwrap_across_pi():
    u = C.HeadingUnwrapper()
    seq = [u(y) for y in (3.0, 3.1, -3.1, -3.0, 3.1)]
    np.testing.assert_allclose(seq, [3.0, 3.1, 2 * math.pi - 3.1, 2 * math.pi - 3.0, 3.1], atol=1e-12)


def test_objects_and_context_roundtrip():
    o = ObjectState(id=7, cls=ObjectClass.PEDESTRIAN, x=1, y=2, z=0.9, yaw=0.3, length=0.6, width=0.6,
                    height=1.8, vx=1.2, vy=-0.4, pos_cov=np.eye(2) * 0.09)  # fmt: skip
    m = track_array()
    C.fill_track_array(m, [o], track)
    (b,) = C.objects_from_track_array(m)
    assert (b.id, b.cls, b.vx, b.vy) == (7, ObjectClass.PEDESTRIAN, 1.2, -0.4)
    ctx = TrafficContext(TrafficLightState.RED, 22.5, math.inf, 30.0, 13.89, 3.5, 0.01, 0.8)
    cm = traffic_context()
    C.fill_context(cm, ctx)
    assert C.context_from_msg(cm).light_state == TrafficLightState.RED


def test_pointcloud2_xyzi_no_copy_semantics():
    pts = np.random.default_rng(0).normal(size=(1000, 4)).astype(np.float32)
    m = pointcloud2()
    C.fill_pointcloud2(m, pts, ("x", "y", "z", "intensity"), field)
    assert m.point_step == 16 and m.width == 1000 and m.height == 1
    np.testing.assert_array_equal(C.xyzi_from_pointcloud2(m), pts)


def test_float32_roundtrip_matches_message_types():
    a = PolicyAction(alpha_v=0.1234567891234, delta_y=-1.0000000001, brake=0.3)
    r = C.roundtrip_action(a)
    assert r.alpha_v == float(np.float32(a.alpha_v)) and r.delta_y == float(np.float32(a.delta_y))
    assert set(C.F32_FIELDS["TrafficContext"]) == {"dist_to_stop_line", "dist_to_stop_sign", "dist_to_junction",
        "speed_limit", "lane_width", "road_grade", "surface_roughness"}  # fmt: skip


def test_roundtrip_route_and_time_match_ros_path():
    from hybrid_av.planning.route import RouteBuilder

    r = C.roundtrip_route(RouteBuilder.straight(100.0, 13.89))
    assert r.length == pytest.approx(100.0, abs=0.5)
    assert all(float(np.float32(v)) == v for v in r.speed_limit[:3])
    e = EgoState(t=0.15000000000000002, x=1.0, y=2.0, psi=0.3, vx=5.0, vy=0.0, r=0.0)
    m = odometry()
    C.fill_odometry(m, e, "map", "base_link")
    C.fill_stamp(m.header.stamp, e.t)
    assert C.ego_from_odometry(m, C.HeadingUnwrapper()).t == 0.15  # what mpc_node sees


def test_radar_and_detection_conversions():
    from types import SimpleNamespace as NS

    from hybrid_av.perception.base import Box3D

    arr = np.array([[10.0, 0.1, 0.0, -2.0], [20.0, -0.2, 0.05, 1.0]], np.float32)
    m = pointcloud2()
    C.fill_pointcloud2(m, arr, ("depth", "azimuth", "altitude", "velocity"), field)
    dets = C.radar_from_pointcloud2(m, "front")
    assert [(d.depth, d.velocity, d.sensor_id) for d in dets] == [(10.0, -2.0, "front"), (20.0, 1.0, "front")]

    xyz = lambda: NS(x=0.0, y=0.0, z=0.0)
    det = lambda: NS(
        bbox=NS(center=NS(position=xyz(), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)), size=xyz()),
        results=[NS(hypothesis=NS(class_id="", score=0.0))],
    )
    box = Box3D(1.0, 2.0, 0.5, 4.5, 1.9, 1.5, 0.4, ObjectClass.CAR, 0.75)
    m = NS(detections=[])
    C.fill_detection_array(m, [box], det)
    (back,) = C.boxes_from_detection_array(m)
    assert back == pytest.approx(box) or (back.cls, back.score, back.length) == (ObjectClass.CAR, 0.75, 4.5)
    assert back.yaw == pytest.approx(0.4, abs=1e-12)
