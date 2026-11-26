import math

import numpy as np
import pytest

from hybrid_av.geometry import (
    box_corners,
    iou_3d,
    iou_bev,
    obb_overlap,
    rotate,
    to_body,
    to_world,
    unwrap_near,
    unwrap_sequence,
    wrap_angle,
)
from hybrid_av.perception.base import Box3D
from hybrid_av.types import ObjectClass


def test_wrap_angle_seam():
    assert wrap_angle(np.pi) == np.pi
    assert wrap_angle(-np.pi) == np.pi
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    np.testing.assert_allclose(wrap_angle(np.array([0.0, 7.0])), [0.0, 7.0 - 2 * np.pi])


def test_unwrap_near():
    assert np.isclose(unwrap_near(0.1, 4 * np.pi), 4 * np.pi + 0.1)
    assert np.isclose(unwrap_near(-3.1, 3.1), 2 * np.pi - 3.1)


def test_unwrap_sequence_two_laps():
    a = wrap_angle(np.linspace(0, 4 * np.pi, 400))
    u = unwrap_sequence(a)
    assert np.isclose(u[-1], 4 * np.pi) and np.all(np.diff(u) > 0)


def test_body_world_roundtrip():
    p = np.array([[3.0, -1.0], [0.0, 2.0]])
    np.testing.assert_allclose(to_world(to_body(p, 1.0, 2.0, 0.7), 1.0, 2.0, 0.7), p, atol=1e-12)
    np.testing.assert_allclose(to_body(np.array([1.0, 1.0]), 0.0, 0.0, np.pi / 2), [1.0, -1.0], atol=1e-12)


def test_obb_overlap_axis_aligned():
    assert obb_overlap((0, 0, 0, 4, 2), (3.9, 0, 0, 4, 2))
    assert obb_overlap((0, 0, 0, 4, 2), (4.0, 0, 0, 4, 2))  # touching
    assert not obb_overlap((0, 0, 0, 4, 2), (4.01, 0, 0, 4, 2))


def test_obb_rotated_near_miss():
    # B's facing edge is x + y = 2.28; A's nearest corner (1, 1) has x + y = 2, so no touch
    assert not obb_overlap((0, 0, 0, 2, 2), (2.2, 2.2, np.pi / 4, 3, 3))
    assert obb_overlap((0, 0, 0, 2, 2), (1.9, 1.9, np.pi / 4, 3, 3))  # facing edge 1.68 < 2


def test_box_corners_ccw():
    c = box_corners(0, 0, 0, 4, 2)
    assert c.shape == (4, 2)
    x, y = c[:, 0], c[:, 1]
    assert np.isclose(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y), 8.0)
    c = box_corners(0, 0, np.pi / 2, 4, 2)
    assert np.isclose(c[:, 0].max(), 1.0) and np.isclose(c[:, 0].min(), -1.0)


def test_rotate_shapes():
    assert rotate(np.array([1.0, 0.0]), 0.3).shape == (2,)
    assert rotate(np.zeros((5, 2)), 0.3).shape == (5, 2)
    np.testing.assert_allclose(rotate(np.array([1.0, 0.0]), np.pi / 2), [0.0, 1.0], atol=1e-12)


def test_base_world_roundtrip():
    import math

    from hybrid_av.geometry import base_to_world, world_to_base
    from hybrid_av.types import EgoState

    e = EgoState(t=0, x=10, y=-5, psi=math.pi / 2, vx=0, vy=0, r=0)
    np.testing.assert_allclose(base_to_world(np.array([[1.0, 0.0]]), e), [[10.0, -4.0]], atol=1e-12)
    pts = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(world_to_base(base_to_world(pts, e), e), pts, atol=1e-12)


def B(x=0, y=0, z=0, l=4, w=2, h=2, yaw=0.0):
    return Box3D(x=x, y=y, z=z, length=l, width=w, height=h, yaw=yaw, cls=ObjectClass.CAR, score=1.0)


def test_iou_identical_and_disjoint():
    assert iou_3d(B(), B()) == pytest.approx(1.0)
    assert iou_3d(B(), B(x=10)) == 0.0


def test_iou_half_length_shift():
    assert iou_bev(B(), B(x=2)) == pytest.approx(1 / 3)


def test_iou_rotation_90_degrees():
    # 4x2 vs 2x4 at the same centre: intersection 2x2 = 4, union 8 + 8 - 4 = 12
    assert iou_bev(B(), B(yaw=math.pi / 2)) == pytest.approx(1 / 3)


def test_iou_3d_vertical_overlap():
    assert iou_3d(B(), B(z=1)) == pytest.approx(1 / 3)
