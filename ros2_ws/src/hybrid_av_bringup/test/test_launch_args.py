"""stack.launch.py arguments and the LiteSim closed loop through it (ROS dev container only)."""

import json
import subprocess
import time

import pytest

pytestmark = pytest.mark.ros
LAUNCH = ["ros2", "launch", "hybrid_av_bringup", "stack.launch.py"]
NODES = {"litesim_node", "mpc_node", "policy_node", "localization_node", "static_tf"}


def test_litesim_rejects_perception_mode():
    r = subprocess.run(
        [*LAUNCH, "sim:=litesim", "mode:=perception", "foxglove:=false"],
        check=False, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert r.returncode != 0
    assert "perception mode needs sim:=carla" in r.stdout + r.stderr


def test_litesim_ground_truth_runs_and_exits(tmp_path):
    import rclpy
    import tf2_ros
    from rclpy.time import Time

    proc = subprocess.Popen(
        [*LAUNCH, "sim:=litesim", "mode:=ground_truth", "policy:=constant", "foxglove:=false", "max_ticks:=40",
         f"result_path:={tmp_path / 'result.json'}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    rclpy.init()
    n = rclpy.create_node("launch_probe")
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, n, spin_thread=False)
    names, found = set(), False
    end = time.monotonic() + 120
    while time.monotonic() < end and proc.poll() is None and not (found and NODES <= names):
        rclpy.spin_once(n, timeout_sec=0.1)
        names |= {nm for nm, _ in n.get_node_names_and_namespaces()}
        found = found or buf.can_transform("base_link", "lidar_top", Time())
    try:
        code = proc.wait(timeout=120)  # ends by itself after 40 ticks
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("launch did not exit after max_ticks")
    out = proc.stdout.read()
    n.destroy_node()
    rclpy.shutdown()
    assert NODES <= names, names
    assert found, "no TF base_link -> lidar_top"
    assert code == 0, out[-2000:]
    stats = json.loads((tmp_path / "result.json").read_text())["tick_stats"]
    assert stats["ticks"] == 40 and stats["warmup_ticks"] == 0 and stats["late_commands"] == 0, (
        stats,
        out[-2000:],
    )
