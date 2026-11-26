"""Every CARLA <-> library frame and sign convention (simulation-backends.md §2). Pure: never imports carla.

CARLA is left-handed (y right, yaw clockwise, degrees); the library is right-handed (y left, yaw CCW, radians).
CARLA objects arrive as duck-typed attribute bags.
"""

import math

import numpy as np

from hybrid_av.geometry import world_to_base, wrap_angle
from hybrid_av.perception.base import Box3D, RadarDetection
from hybrid_av.perception.classes import CARLA_ACTOR_TO_CLASS
from hybrid_av.types import EgoState, ObjectClass, ObjectState


def loc_to_lib(x, y, z):
    # ECC-04: position (x, -y, z)
    return (x, -y, z)


lib_to_loc = loc_to_lib  # the flip is its own inverse


def yaw_to_lib(yaw_deg):
    # ECC-04: CARLA yaw is clockwise degrees; library psi is CCW radians
    return -math.radians(yaw_deg)


def lib_to_yaw(psi):
    return -math.degrees(psi)


def vel_to_lib(vx, vy, vz):
    # ECC-04: world velocity (vx, -vy, vz)
    return (vx, -vy, vz)


def angvel_z_to_lib(wz_deg_s):
    # ECC-04: CARLA reports angular velocity in deg/s, clockwise positive
    return -math.radians(wz_deg_s)


def body_velocity(vx_w, vy_w, psi):
    c, s = math.cos(psi), math.sin(psi)
    return (c * vx_w + s * vy_w, -s * vx_w + c * vy_w)


def ego_from_actor(transform, velocity, angular_velocity, acceleration, t, unwrap) -> EgoState:
    x, y, _ = loc_to_lib(transform.location.x, transform.location.y, transform.location.z)
    psi = unwrap(yaw_to_lib(transform.rotation.yaw))
    vxw, vyw, _ = vel_to_lib(velocity.x, velocity.y, velocity.z)
    vx, vy = body_velocity(vxw, vyw, psi)
    axw, ayw, _ = vel_to_lib(acceleration.x, acceleration.y, acceleration.z)
    ax, _ = body_velocity(axw, ayw, psi)
    return EgoState(t=t, x=x, y=y, psi=psi, vx=vx, vy=vy, r=angvel_z_to_lib(angular_velocity.z), ax=ax)


def object_from_actor(
    actor_id, type_id, transform, bbox_extent, bbox_location, velocity, number_of_wheels=4
) -> ObjectState:
    x, y, z = loc_to_lib(transform.location.x, transform.location.y, transform.location.z)
    yaw = yaw_to_lib(transform.rotation.yaw)
    # the bbox centre is offset from the actor origin in the actor frame
    ox, oy = bbox_location.x, -bbox_location.y
    c, s = math.cos(yaw), math.sin(yaw)
    vx, vy, _ = vel_to_lib(velocity.x, velocity.y, velocity.z)
    cls = CARLA_ACTOR_TO_CLASS(type_id)
    # ECC-25: extents are half-sizes (length = 2*extent.x); two-wheeled vehicle.* is a cyclist
    if type_id.startswith("vehicle.") and number_of_wheels == 2:
        cls = ObjectClass.CYCLIST
    return ObjectState(
        id=actor_id,
        cls=cls,
        x=x + c * ox - s * oy,
        y=y + s * ox + c * oy,
        z=z + bbox_location.z,
        yaw=yaw,
        length=2 * bbox_extent.x,
        width=2 * bbox_extent.y,
        height=2 * bbox_extent.z,
        vx=vx,
        vy=vy,
    )


def lidar_to_lib(raw: np.ndarray) -> np.ndarray:
    # ECC-08: sensor-frame points need only y -> -y; ECC-09: (N*4,) float32 buffer, intensity is the 4th float
    pts = raw.reshape(-1, 4).copy()
    pts[:, 1] *= -1
    return pts


def radar_to_lib(raw: np.ndarray, sensor_id) -> list[RadarDetection]:
    # ECC-10: layout [velocity, azimuth, altitude, depth] float32, azimuth negated
    return [
        RadarDetection(
            depth=float(d), azimuth=-float(az), altitude=float(al), velocity=float(v), sensor_id=sensor_id
        )
        for v, az, al, d in raw.reshape(-1, 4)
    ]


def imu_to_lib(accel, gyro) -> tuple[np.ndarray, np.ndarray]:
    # ECC-06: accelerometer (ax, -ay, az), gyroscope (-gx, gy, -gz); signs unverified (CARLA issue #9563)
    return (
        np.array([accel.x, -accel.y, accel.z], dtype=np.float64),
        np.array([-gyro.x, gyro.y, -gyro.z], dtype=np.float64),
    )


def mount_to_carla(x, y, z, yaw):
    return (x, -y, z), lib_to_yaw(yaw)


def bbox_to_box3d(actor_transform, bbox_extent, bbox_location, ego: EgoState, cls) -> Box3D:
    o = object_from_actor(0, "", actor_transform, bbox_extent, bbox_location, _ZERO, number_of_wheels=4)
    x, y = world_to_base(np.array([o.x, o.y]), ego)
    return Box3D(
        x=float(x), y=float(y), z=o.z, length=o.length, width=o.width, height=o.height,
        yaw=wrap_angle(o.yaw - ego.psi), cls=cls, score=1.0,
    )  # fmt: skip


class _Zero:
    x = y = z = 0.0


_ZERO = _Zero()


def steer_from_delta(delta, max_steer_rad) -> float:
    # ECC-05: CARLA steer is right-positive, library delta is left-positive
    return float(np.clip(-delta / max_steer_rad, -1.0, 1.0))
