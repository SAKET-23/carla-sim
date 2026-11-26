"""Angle wrapping, 2-D frame transforms and oriented-box overlap (SI units, radians)."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # perception.base imports this module
    from hybrid_av.perception.base import Box3D

_TWO_PI = 2.0 * np.pi
_TOUCH_TOL = 1e-9


def wrap_angle(a):
    """Wrap into (-pi, pi]; both +-pi map to +pi."""
    r = np.pi - ((np.pi - np.asarray(a, dtype=np.float64)) % _TWO_PI)
    return float(r) if r.ndim == 0 else r


def unwrap_near(a: float, ref: float) -> float:
    """a + 2*pi*k with |result - ref| <= pi."""
    return float(a + _TWO_PI * np.round((ref - a) / _TWO_PI))


def unwrap_sequence(a: np.ndarray) -> np.ndarray:
    return np.unwrap(a)


def rotate(xy: np.ndarray, theta: float) -> np.ndarray:
    """Rotate (2,) or (N,2) points CCW by theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(xy, dtype=np.float64) @ np.array([[c, s], [-s, c]])


def to_body(xy_world: np.ndarray, x: float, y: float, psi: float) -> np.ndarray:
    return rotate(np.asarray(xy_world, dtype=np.float64) - np.array([x, y]), -psi)


def to_world(xy_body: np.ndarray, x: float, y: float, psi: float) -> np.ndarray:
    return rotate(xy_body, psi) + np.array([x, y])


def box_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    """(4,2) corners, CCW: front-left, rear-left, rear-right, front-right."""
    hl, hw = length / 2, width / 2
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return to_world(local, x, y, yaw)


def obb_overlap(a, b) -> bool:
    """Separating-axis test on edge normals; touching counts as overlap."""
    ca, cb = box_corners(*a), box_corners(*b)
    for c in (ca, cb):
        for edge in np.roll(c, -1, axis=0) - c:
            n = np.array([-edge[1], edge[0]])
            pa, pb = ca @ n, cb @ n
            if pa.max() < pb.min() - _TOUCH_TOL * np.linalg.norm(
                n
            ) or pb.max() < pa.min() - _TOUCH_TOL * np.linalg.norm(n):
                return False
    return True


def base_to_world(xy: np.ndarray, ego) -> np.ndarray:
    """(N,2) base_link points to the world frame using an EgoState."""
    return to_world(xy, ego.x, ego.y, ego.psi)


def world_to_base(xy: np.ndarray, ego) -> np.ndarray:
    return to_body(xy, ego.x, ego.y, ego.psi)


def box_corners_bev(x: float, y: float, l: float, w: float, yaw: float) -> np.ndarray:
    """(4,2) corners, counter-clockwise."""
    return box_corners(x, y, yaw, l, w)


def polygon_intersection_area(p: np.ndarray, q: np.ndarray) -> float:
    """Area of the intersection of two convex counter-clockwise polygons (Sutherland-Hodgman)."""
    out = [tuple(v) for v in p]
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        if not out:
            return 0.0
        inp, out = out, []
        side = [(b[0] - a[0]) * (v[1] - a[1]) - (b[1] - a[1]) * (v[0] - a[0]) for v in inp]  # >= 0: inside
        for i, v in enumerate(inp):
            j = i - 1
            if side[i] >= 0:
                if side[j] < 0:
                    out.append(_cross_point(inp[j], v, side[j], side[i]))
                out.append(v)
            elif side[j] >= 0:
                out.append(_cross_point(inp[j], v, side[j], side[i]))
    if len(out) < 3:
        return 0.0
    xs, ys = np.array(out).T
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2)


def _cross_point(u, v, su, sv):
    t = su / (su - sv)
    return (u[0] + t * (v[0] - u[0]), u[1] + t * (v[1] - u[1]))


def _bev_inter(a: "Box3D", b: "Box3D") -> float:
    return polygon_intersection_area(
        box_corners_bev(a.x, a.y, a.length, a.width, a.yaw),
        box_corners_bev(b.x, b.y, b.length, b.width, b.yaw),
    )


def iou_bev(a: "Box3D", b: "Box3D") -> float:
    inter = _bev_inter(a, b)
    union = a.length * a.width + b.length * b.width - inter
    return inter / union if union > 0 else 0.0


def iou_3d(a: "Box3D", b: "Box3D") -> float:
    """z is the box centre."""
    dz = min(a.z + a.height / 2, b.z + b.height / 2) - max(a.z - a.height / 2, b.z - b.height / 2)
    inter = _bev_inter(a, b) * max(dz, 0.0)
    union = a.length * a.width * a.height + b.length * b.width * b.height - inter
    return inter / union if union > 0 else 0.0
