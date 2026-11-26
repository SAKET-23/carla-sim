"""In-process vs ROS parity on LiteSim through stack.launch.py (ROS dev container only)."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.ros
sys.path.insert(0, str(Path(__file__).parent))  # parity_reference sits next to this file
N = 200
LAUNCH = ["ros2", "launch", "hybrid_av_bringup", "stack.launch.py"]


@pytest.fixture(scope="module")
def policy_path(tmp_path_factory):
    import torch

    from hybrid_av.policy.io import save_policy
    from hybrid_av.policy.network import CompactPolicy

    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("policy") / "random.pt"
    save_policy(CompactPolicy(), p, {"note": "random weights for the parity test"})
    return p


def record_commands(policy_path, tmp_path, n=N):
    """Launch the stack and return (commands (n, 3) in stamp order, litesim result json)."""
    import json

    import rclpy
    from hybrid_av_msgs.msg import VehicleCommand
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    result = tmp_path / "result.json"
    rclpy.init()
    node = rclpy.create_node("parity_recorder")
    got: dict[int, tuple[float, float, float]] = {}

    def on_cmd(m):
        got[m.header.stamp.sec * 10**9 + m.header.stamp.nanosec] = (m.steer, m.throttle, m.brake)

    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1000)
    node.create_subscription(VehicleCommand, "/carla/ego/command", on_cmd, qos)
    proc = subprocess.Popen(
        [*LAUNCH, "sim:=litesim", "mode:=ground_truth", "scenario:=nominal", "seed:=0",
         f"policy:={policy_path}", "foxglove:=false", f"max_ticks:={n}", f"result_path:={result}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    end = time.monotonic() + 600
    while time.monotonic() < end and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
    out = proc.communicate(timeout=60)[0]
    node.destroy_node()
    rclpy.shutdown()
    assert proc.returncode == 0, out[-3000:]
    assert "late policy action" not in out, out[-3000:]
    return np.array([got[k] for k in sorted(got)]), json.loads(result.read_text())


def test_parity_litesim_200_steps(policy_path, tmp_path):
    from parity_reference import reference_commands

    recorded, summary = record_commands(policy_path, tmp_path)
    ref = reference_commands("nominal", 0, policy_path, n=N)
    assert recorded.shape == (N, 3)
    assert summary["tick_stats"]["late_commands"] == 0
    diff = np.abs(recorded - ref)
    print(f"parity max abs diff (steer, throttle, brake): {diff.max(axis=0)}")
    k = int(np.argmax(diff.max(axis=1) > 1e-6)) if (diff > 1e-6).any() else -1
    assert diff.max() <= 1e-6, (
        f"first diverging step {k}: ros {recorded[k]} ref {ref[k]}; max {diff.max(axis=0)}"
    )
