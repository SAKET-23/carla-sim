"""IC-11...IC-16: sensors, synchronisation and extrinsics (integration-check.md sections B and C)."""

import math
import time

import numpy as np
import pytest
import yaml

from hybrid_av.sim.carla_backend import convert
from hybrid_av.sim.carla_backend.sensors import load_rig
from tests.integration.helpers import RIG, Driver, spawn_tracked, straight_sim

pytestmark = pytest.mark.carla


def _lib_offset(ego_tf, x: float, y: float):
    """CARLA location of the library body-frame offset (x forward, y left) from the ego."""
    import carla

    psi = convert.yaw_to_lib(ego_tf.rotation.yaw)
    lx, ly, _ = convert.loc_to_lib(ego_tf.location.x, ego_tf.location.y, 0.0)
    wx, wy = lx + x * math.cos(psi) - y * math.sin(psi), ly + x * math.sin(psi) + y * math.cos(psi)
    return carla.Location(wx, -wy, ego_tf.location.z + 0.5)


def _step_idle(sim, obs, n=3):
    for _ in range(n):
        obs, _ = Driver(sim, 0.0).step(obs)
    return obs


def test_lidar_left_prop(make_sim, points):
    """IC-11 — see Plan/06-integration/integration-check.md; likely fix: ECC-08."""
    import carla

    sim, obs = straight_sim(make_sim, points, sensors=RIG)
    spawn_tracked(
        sim, "static.prop.streetbarrier", carla.Transform(_lib_offset(sim.ego.get_transform(), 10.0, 4.0))
    )
    _step_idle(sim, obs)
    pts = convert.lidar_to_lib(np.frombuffer(sim.last_sensor_data["lidar"].raw_data, np.float32))
    z = pts[:, 2] + 1.9  # lidar mount height, base_link frame
    near = lambda cy: pts[(z > 0.2) & (np.hypot(pts[:, 0] - 10.0, pts[:, 1] - cy) < 2.0)]
    left, right = near(4.0), near(-4.0)
    assert len(left) > 5 and np.hypot(left[:, 0].mean() - 10.0, left[:, 1].mean() - 4.0) < 0.5
    assert len(right) == 0


def test_lidar_full_sweep(make_sim, points):
    """IC-12 — see Plan/06-integration/integration-check.md; likely fix: ECC-09."""
    sim, obs = straight_sim(make_sim, points, sensors=RIG)
    counts = []
    for _ in range(20):
        obs, _ = Driver(sim, 0.0).step(obs)
        pts = convert.lidar_to_lib(np.frombuffer(sim.last_sensor_data["lidar"].raw_data, np.float32))
        az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        assert len(np.unique(np.floor(az))) > 350  # azimuth coverage in 1 degree bins
        counts.append(len(pts))
    print(f"LiDAR points per tick: mean {np.mean(counts):.0f} (65,000 minus sky misses)")
    assert (
        0.4 * 65_000 < np.mean(counts) <= 1.1 * 65_000
    )  # shortcut: the miss count is not known, so a loose floor


def test_radar_signs(make_sim, points):
    """IC-13 — see Plan/06-integration/integration-check.md; likely fix: ECC-10."""
    import carla

    sim, obs = straight_sim(make_sim, points, sensors=RIG)
    ego_tf = sim.ego.get_transform()
    start = _lib_offset(ego_tf, 30.0, 3.0)
    target = spawn_tracked(
        sim, "vehicle.audi.tt", carla.Transform(start, carla.Rotation(yaw=ego_tf.rotation.yaw + 180.0))
    )
    toward = carla.Vector3D(start.x - ego_tf.location.x, start.y - ego_tf.location.y, 0.0)
    norm = math.hypot(toward.x, toward.y)
    azs, ratio = [], []
    for _ in range(8):
        target.set_target_velocity(carla.Vector3D(-5.0 * toward.x / norm, -5.0 * toward.y / norm, 0.0))
        obs, _ = Driver(sim, 0.0).step(obs)
        dets = convert.radar_to_lib(
            np.frombuffer(sim.last_sensor_data["radar_front"].raw_data, np.float32), "front"
        )
        tl = target.get_transform().location
        dist = math.hypot(tl.x - ego_tf.location.x - 2.4, tl.y - ego_tf.location.y)
        for d in dets:
            # moving returns only: static clutter at the same depth reads 0 m/s (pod probe: target -5.0, clutter 0.0)
            if abs(d.depth - dist) < 2.0 and d.azimuth > 0.0 and abs(d.velocity) > 1.0:
                azs.append(d.azimuth)
                ratio.append(abs(abs(d.velocity) - 5.0 * math.cos(d.azimuth)) if d.velocity < 0 else 99.0)
    assert azs, "no front-radar detection on the target (azimuth > 0)"
    assert np.mean(ratio) < 0.5


def test_frame_alignment(make_sim, points):
    """IC-14 — see Plan/06-integration/integration-check.md; likely fix: ECC-11."""
    sim, obs = straight_sim(make_sim, points, sensors=RIG)
    drv, worst = Driver(sim, 0.0), 0.0
    for _ in range(300):
        t0 = time.monotonic()
        obs, _ = drv.step(obs)
        worst = max(worst, time.monotonic() - t0)
        for sid, d in sim.last_sensor_data.items():
            assert d is not None, f"{sid} missing at frame {sim.last_frame}"
            assert (d[-1] if isinstance(d, list) else d).frame == sim.last_frame
    assert not sim._queues.missing or sum(sim._queues.missing.values()) == 0
    print(f"max step wall time {worst:.3f} s")  # includes the sensor wait


def test_camera_projection(make_sim, points):
    """IC-15 — see Plan/06-integration/integration-check.md; likely fix: ECC-12."""
    import carla

    sim, obs = straight_sim(make_sim, points, sensors=RIG)
    ego_tf = sim.ego.get_transform()
    target = spawn_tracked(
        sim, "vehicle.audi.tt", carla.Transform(_lib_offset(ego_tf, 15.0, 0.0), ego_tf.rotation)
    )
    target.apply_control(carla.VehicleControl(hand_brake=True))
    _step_idle(sim, obs, 5)
    spec = next(s for s in load_rig(RIG) if s.id == "camera_front")
    w, h, fov = (
        int(spec.attributes["image_size_x"]),
        int(spec.attributes["image_size_y"]),
        float(spec.attributes["fov"]),
    )
    f = w / (2 * math.tan(math.radians(fov) / 2))
    tf, box = target.get_transform(), target.bounding_box
    ego_tf = sim.ego.get_transform()
    psi_e = convert.yaw_to_lib(ego_tf.rotation.yaw)
    ex, ey, _ = convert.loc_to_lib(ego_tf.location.x, ego_tf.location.y, ego_tf.location.z)
    psi_t = convert.yaw_to_lib(tf.rotation.yaw)
    tx, ty, tz = convert.loc_to_lib(tf.location.x, tf.location.y, tf.location.z)
    us, vs = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                lx, ly, lz = (
                    sx * box.extent.x + box.location.x,
                    -(sy * box.extent.y + box.location.y),
                    sz * box.extent.z + box.location.z,
                )
                wx = tx + lx * math.cos(psi_t) - ly * math.sin(psi_t)
                wy = ty + lx * math.sin(psi_t) + ly * math.cos(psi_t)
                dx, dy = wx - ex, wy - ey
                bx = (
                    dx * math.cos(psi_e) + dy * math.sin(psi_e) - spec.x
                )  # body frame, relative to the camera
                by = -dx * math.sin(psi_e) + dy * math.cos(psi_e) - spec.y
                bz = tz + lz - ego_tf.location.z - spec.z
                us.append(f * (-by) / bx + w / 2)  # optical frame: x right = -y, y down = -z, z forward = x
                vs.append(f * (-bz) / bx + h / 2)
    proj = (max(min(us), 0), max(min(vs), 0), min(max(us), w), min(max(vs), h))
    sem = sim.last_sensor_data["camera_front_semantic"]
    tags = np.frombuffer(sem.raw_data, np.uint8).reshape(sem.height, sem.width, 4)[:, :, 2]  # ECC-31
    mask = np.isin(tags, (14, 15, 16))
    pad_u, pad_v = 0.2 * (proj[2] - proj[0]), 0.2 * (proj[3] - proj[1])
    region = np.zeros_like(mask)
    region[
        int(max(proj[1] - pad_v, 0)) : int(proj[3] + pad_v),
        int(max(proj[0] - pad_u, 0)) : int(proj[2] + pad_u),
    ] = True
    vs_idx, us_idx = np.nonzero(mask & region)
    assert len(us_idx), "no vehicle-tagged pixel in the projected region"
    gt = (us_idx.min(), vs_idx.min(), us_idx.max() + 1, vs_idx.max() + 1)
    iw, ih = min(proj[2], gt[2]) - max(proj[0], gt[0]), min(proj[3], gt[3]) - max(proj[1], gt[1])
    inter = max(iw, 0) * max(ih, 0)
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    assert inter / (area(proj) + area(gt) - inter) >= 0.6


def test_sensor_transforms(make_sim, points):
    """IC-16 — see Plan/06-integration/integration-check.md; likely fix: ECC-12. (calibration.md section 4)"""
    sim, _ = straight_sim(make_sim, points, sensors=RIG)
    specs = yaml.safe_load(RIG.read_text())["sensors"]
    attached = [
        a for a in sim._session.actors if a.type_id.startswith("sensor.") and "collision" not in a.type_id
    ]
    assert len(attached) == len(specs)
    ego_tf = sim.ego.get_transform()
    # full inverse (roll/pitch too): a yaw-only frame put a 0.5 deg road pitch into x/z (1.4 cm, pod G2)
    inv = np.array(ego_tf.get_inverse_matrix())
    for actor, spec in zip(attached, specs, strict=True):
        tf, m = actor.get_transform(), spec["mount"]
        rx, ry, rz, _ = inv @ np.array([tf.location.x, tf.location.y, tf.location.z, 1.0])
        lib = (rx, -ry, rz)
        yaw = convert.yaw_to_lib(tf.rotation.yaw - ego_tf.rotation.yaw)
        assert lib == pytest.approx((m["x"], m["y"], m["z"]), abs=0.01), spec["id"]
        d = (yaw - m["yaw"] + math.pi) % (2 * math.pi) - math.pi
        assert abs(d) < math.radians(0.5), spec["id"]
