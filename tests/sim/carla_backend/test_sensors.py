import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from hybrid_av.sim.carla_backend.client import CarlaSimConfig, carla_session
from hybrid_av.sim.carla_backend.sensors import SensorQueues, load_rig, spawn_rig

_ROOT = Path(__file__).resolve().parents[3]
RIG = _ROOT / "configs/sensors/carla_rig.yaml"


def d(frame):
    return NS(frame=frame)


def test_queues_match_frame_and_time_out():
    q = SensorQueues(["cam", "lidar", "imu"])
    cam, lidar, imu = q.callback("cam"), q.callback("lidar"), q.callback("imu")
    cam(d(9))
    cam(d(10))
    lidar(d(11))
    imu(d(10))
    imu(d(10))
    out = q.collect(10, timeout=0.05)
    assert out["cam"].frame == 10 and out["lidar"] is None and q.missing["lidar"] == 1
    assert [x.frame for x in out["imu"]] == [10, 10]
    lidar(d(12))
    cam(d(11))
    assert q.collect(11, timeout=0.05)["lidar"].frame == 11  # future frame was kept


def test_collect_waits_for_a_late_callback():
    import threading

    q = SensorQueues(["cam"])
    threading.Timer(0.05, lambda: q.callback("cam")(d(3))).start()
    assert q.collect(3, timeout=2.0)["cam"].frame == 3 and q.missing["cam"] == 0


def test_rig_matches_simulation_backends_table():
    specs = {s.id: s for s in load_rig(RIG)}
    assert (
        specs["lidar"].attributes["channels"] == "64"
        and specs["lidar"].attributes["rotation_frequency"] == "20"
    )
    assert specs["lidar"].attributes["points_per_second"] == "1300000"
    assert {k for k in specs if k.startswith("radar_")} == {
        "radar_front",
        "radar_rear",
        "radar_left",
        "radar_right",
    }
    assert specs["imu"].attributes["sensor_tick"] == "0.01"
    assert all(s.attributes.get("sensor_tick", "0.0") == "0.0" for k, s in specs.items() if k != "imu")


def test_rig_radars_agree_with_phase6_rig_py():
    path = _ROOT / "ros2_ws/src/hybrid_av_nodes/hybrid_av_nodes/rig.py"
    if not path.exists():
        pytest.skip("Phase 6 not present")
    mod = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("rig6", path))
    spec.loader.exec_module(mod)
    specs = {s.id.removeprefix("radar_"): s for s in load_rig(RIG) if s.id.startswith("radar_")}
    for r in mod.load_rig(RIG).radars:
        s = specs[r.sensor_id]
        assert (s.x, s.y, s.z, s.yaw) == pytest.approx((r.x, r.y, r.z, r.yaw))


def test_spawn_rig_attaches_converts_mounts_and_delivers_by_frame(fake_carla):
    import carla

    with carla_session(CarlaSimConfig()) as s:
        bp_lib = s.world.get_blueprint_library()
        ego = s.world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), carla.Transform())
        specs = load_rig(RIG)
        q = SensorQueues([x.id for x in specs])
        sensors = dict(zip((x.id for x in specs), spawn_rig(s, ego, specs, q), strict=True))
        assert all(x.attach_to is ego and x.is_listening() for x in sensors.values())
        assert all(x in s.actors for x in sensors.values())
        left = sensors["radar_left"].get_transform()  # library y 1.0, yaw +90 deg -> CARLA y -1.0, yaw -90
        assert (left.location.y, left.rotation.yaw) == pytest.approx((-1.0, -90.0))
        assert sensors["lidar"].attributes["channels"] == "64"
        s.world.inject(sensors["lidar"].id, carla.LidarMeasurement())
        s.world.inject(sensors["radar_front"].id, carla.RadarMeasurement())
        s.world.drop_sensor(sensors["radar_front"].id, s.world.get_snapshot().frame + 1)
        frame = s.world.tick()
        out = q.collect(frame, timeout=0.05)
        assert out["lidar"].frame == frame and out["radar_front"] is None and q.missing["radar_front"] == 1
        assert math.isclose(sensors["lidar"].get_transform().location.z, 1.9)
