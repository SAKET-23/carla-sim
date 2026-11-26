"""Rotation math for static_tf. Pure: no rclpy."""

from math import cos, sin


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """x, y, z, w of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (
        cos(roll / 2),
        sin(roll / 2),
        cos(pitch / 2),
        sin(pitch / 2),
        cos(yaw / 2),
        sin(yaw / 2),
    )
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
