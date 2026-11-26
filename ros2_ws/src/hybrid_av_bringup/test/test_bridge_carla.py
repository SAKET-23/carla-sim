"""IC-29...IC-32: the CARLA bridge and the ground-truth chain (integration-check.md section G). Pod only."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from carla_live import ROOT, launch, need_route, result, spin_until_exit, stamp_ns

pytestmark = pytest.mark.carla


def test_stamps_match_clock(tmp_path):
    """IC-29 — see Plan/06-integration/integration-check.md; likely fix: ECC-20."""
    need_route()
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2

    rclpy.init()
    node = rclpy.create_node("stamp_recorder")
    clocks, seen = [], {}
    node.create_subscription(
        Clock, "/clock", lambda m: clocks.append(m.clock.sec * 10**9 + m.clock.nanosec), 10
    )
    topics = {
        "/carla/ego/camera/front/image": Image, "/carla/ego/camera/front/camera_info": CameraInfo,
        "/carla/ego/lidar/points": PointCloud2, "/carla/ego/radar/front": PointCloud2,
        "/carla/ego/wheel_odom": Odometry, "/carla/ego/ground_truth/odom": Odometry,
    }  # fmt: skip
    for name, typ in topics.items():
        node.create_subscription(
            typ, name, lambda m, n=name: seen.setdefault(n, []).append(stamp_ns(m)), qos_profile_sensor_data
        )
    imu = []
    node.create_subscription(
        Imu, "/carla/ego/imu", lambda m: imu.append(stamp_ns(m)), qos_profile_sensor_data
    )
    proc = subprocess.Popen(
        ["ros2", "launch", "hybrid_av_bringup", "bridge_only.launch.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    end = time.monotonic() + 900
    while time.monotonic() < end and len(clocks) < 200 and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
    proc.terminate()
    out = proc.communicate(timeout=60)[0]
    node.destroy_node()
    rclpy.shutdown()
    assert len(clocks) >= 200, out[-3000:]
    ticks = set(clocks[:200])
    for name, stamps in seen.items():
        assert set(stamps[:150]) <= ticks, (
            f"{name}: stamps that are not a /clock value"
        )  # best effort QoS may drop
    assert set(imu[:300]) <= ticks | {
        t - k * 10_000_000 for t in ticks for k in range(1, 5)
    }  # ECC-07 sub-tick stamps


def test_command_deadline(tmp_path):
    """IC-30 — see Plan/06-integration/integration-check.md; likely fix: ECC-21."""
    need_route()
    import carla
    import rclpy
    from hybrid_av_msgs.msg import VehicleCommand
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock

    out_json = tmp_path / "bridge.json"
    rig = ROOT / "configs/sensors/carla_rig.yaml"
    rclpy.init()
    node = rclpy.create_node("mpc_stand_in")
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
    pub = node.create_publisher(VehicleCommand, "/carla/ego/command", qos)
    state = {"n": 0, "delay_at": None, "control": None}
    world = carla.Client("localhost", 2000).get_world()

    def on_clock(m):
        state["n"] += 1
        cmd = VehicleCommand()
        cmd.header.stamp, cmd.throttle, cmd.steer = m.clock, 0.3, 0.0
        if state["n"] == state["delay_at"]:
            time.sleep(1.5)  # one reply later than the 1.0 s deadline
        pub.publish(cmd)

    node.create_subscription(Clock, "/clock", on_clock, qos)
    # the stand-in answers within 10 ms
    proc = subprocess.Popen(
        ["ros2", "run", "hybrid_av_nodes", "carla_bridge", "--ros-args", "-p", f"sensors:={rig}", "-p", "max_ticks:=600",
         "-p", f"result_path:={out_json}", "-p", "scenario:=nominal", "-p", "seed:=100000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    end = time.monotonic() + 1200
    while time.monotonic() < end and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
    out = proc.communicate(timeout=60)[0]
    assert proc.returncode == 0, out[-3000:]
    assert result(out_json)["tick_stats"]["late_commands"] == 0
    # second run: one injected 1.5 s delay at tick 300 gives exactly one late command, the previous one is reapplied
    state.update(n=0, delay_at=300)
    proc = subprocess.Popen(proc.args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while time.monotonic() < end and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
        if state["n"] == 300 and state["control"] is None:
            ego = next((a for a in world.get_actors().filter("vehicle.tesla.model3")), None)
            state["control"] = ego.get_control().throttle if ego else None
    out = proc.communicate(timeout=60)[0]
    node.destroy_node()
    rclpy.shutdown()
    assert proc.returncode == 0, out[-3000:]
    assert result(out_json)["tick_stats"]["late_commands"] == 1
    assert state["control"] == pytest.approx(0.3, abs=1e-3)  # the previous command stays applied


def test_tf_tree(tmp_path):
    """IC-31 — see Plan/06-integration/integration-check.md; likely fix: ECC-12."""
    need_route()
    import rclpy
    import tf2_ros
    from hybrid_av_nodes.convert import yaw_from_quat
    from rclpy.time import Time

    rclpy.init()
    node = rclpy.create_node("tf_checker")
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)
    proc = launch(
        ["mode:=ground_truth", "policy:=constant", "max_ticks:=100", f"result_path:={tmp_path / 'r.json'}"]
    )
    sensors = yaml.safe_load((ROOT / "configs/sensors/carla_rig.yaml").read_text())["sensors"]
    frames = {
        s["frame"]: s["mount"] for s in sensors
    }  # camera_front is shared by the RGB and semantic camera
    got, end = {}, time.monotonic() + 900
    while time.monotonic() < end and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
        for child in ("base_link", *frames):
            parent = "map" if child == "base_link" else "base_link"
            if child not in got and buf.can_transform(parent, child, Time()):
                got[child] = buf.lookup_transform(parent, child, Time()).transform
    out = spin_until_exit(node, proc)
    node.destroy_node()
    rclpy.shutdown()
    assert "base_link" in got, out[-3000:]
    for frame, m in frames.items():
        t = got[frame]
        assert (t.translation.x, t.translation.y, t.translation.z) == pytest.approx(
            (m["x"], m["y"], m["z"]), abs=1e-3
        ), frame
        assert yaw_from_quat(t.rotation) == pytest.approx(
            np.arctan2(np.sin(m["yaw"]), np.cos(m["yaw"])), abs=1e-3
        ), frame


def test_gt_chain_rates(tmp_path):
    """IC-32 — see Plan/06-integration/integration-check.md; likely fix: ECC-21."""
    need_route()
    import rclpy
    from hybrid_av_msgs.msg import FeatureVector, MpcDiagnostics, PolicyAction, VehicleCommand
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    rclpy.init()
    node = rclpy.create_node("rate_counter")
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1000)
    counts = {"features": 0, "command": 0, "diag": 0, "action": 0}
    for key, typ, topic in (
        ("features", FeatureVector, "/autonomy/features"), ("command", VehicleCommand, "/carla/ego/command"),
        ("diag", MpcDiagnostics, "/autonomy/mpc_diagnostics"), ("action", PolicyAction, "/autonomy/policy_action"),
    ):  # fmt: skip
        node.create_subscription(typ, topic, lambda m, k=key: counts.__setitem__(k, counts[k] + 1), qos)
    res = tmp_path / "r.json"
    proc = launch(["mode:=ground_truth", "policy:=constant", "max_ticks:=300", f"result_path:={res}"])
    out = spin_until_exit(node, proc)
    node.destroy_node()
    rclpy.shutdown()
    assert proc.returncode == 0, out[-3000:]
    for key in ("features", "command", "diag"):
        assert abs(counts[key] - 300) <= 1, counts  # exactly one per tick (tick 0 included)
    assert abs(counts["action"] - 150) <= 2, counts  # one policy action every second tick
    assert result(res)["tick_stats"]["late_commands"] == 0
