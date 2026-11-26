"""IC-34, IC-35: perception nodes and KISS-ICP on a live CARLA stack (integration-check.md section G). Pod only."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from carla_live import launch, need_route, spin_until_exit

pytestmark = pytest.mark.carla
_PERCEPTION = ["mode:=perception", "policy:=constant", "seg_backend:=torch", "det_backend:=torch"]


def test_perception_nodes_publish(tmp_path):
    """IC-34 — see Plan/06-integration/integration-check.md; likely fix: ECC-23/ECC-24."""
    need_route()
    import rclpy
    from hybrid_av_msgs.msg import TrackArray
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection3DArray

    rclpy.init()
    node = rclpy.create_node("perception_counter")
    reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1000)
    counts = dict.fromkeys(("seg", "det", "tracks", "kiss", "ego"), 0)
    for key, typ, topic, qos in (
        ("seg", Image, "/perception/segmentation", qos_profile_sensor_data),
        ("det", Detection3DArray, "/perception/detections3d", reliable),
        ("tracks", TrackArray, "/perception/tracks3d", reliable),
        ("kiss", Odometry, "/kiss/odometry", 10),
        ("ego", Odometry, "/localization/ego_state", reliable),
    ):
        node.create_subscription(typ, topic, lambda m, k=key: counts.__setitem__(k, counts[k] + 1), qos)
    proc = launch([*_PERCEPTION, "max_ticks:=600", f"result_path:={tmp_path / 'r.json'}"])
    out = spin_until_exit(node, proc, timeout=3000)
    node.destroy_node()
    rclpy.shutdown()
    assert proc.returncode == 0, out[
        -3000:
    ]  # no node crashed: the launch ends with the bridge, not a failure
    assert "process has died" not in out, out[-3000:]
    for key, n in counts.items():
        assert 590 <= n <= 601, f"{key}: {n} messages for 600 ticks"  # accuracy is G3's job


def test_kiss_icp_sanity(tmp_path):
    """IC-35 — see Plan/06-integration/integration-check.md; likely fix: ECC-24."""
    need_route()
    import rclpy
    import tf2_ros
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.time import Time

    rclpy.init()
    node = rclpy.create_node("kiss_checker")
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)
    gt, est, travelled = [], [], [0.0]
    reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1000)

    def on_gt(m):
        p = (m.pose.pose.position.x, m.pose.pose.position.y)
        if gt:
            travelled[0] += float(np.hypot(p[0] - gt[-1][0], p[1] - gt[-1][1]))
        gt.append(p)

    node.create_subscription(Odometry, "/carla/ego/ground_truth/odom", on_gt, reliable)
    proc = launch([*_PERCEPTION, "max_ticks:=4000", f"result_path:={tmp_path / 'r.json'}"])
    end, frames_ok = time.monotonic() + 3600, False
    while time.monotonic() < end and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
        if buf.can_transform("map", "base_link", Time()):
            t = buf.lookup_transform("map", "base_link", Time()).transform.translation
            est.append((t.x, t.y))
        frames_ok = frames_ok or (
            buf.can_transform("map", "odom", Time()) and buf.can_transform("odom", "base_link", Time())
        )
    spin_until_exit(node, proc, timeout=60)
    node.destroy_node()
    rclpy.shutdown()
    assert frames_ok and est and gt
    if travelled[0] < 400.0:
        pytest.skip(f"the route is only {travelled[0]:.0f} m; IC-35 needs about 500 m")
    err = float(np.hypot(est[-1][0] - gt[-1][0], est[-1][1] - gt[-1][1]))
    assert err < 0.02 * travelled[0] and err < 10.0, (
        f"KISS-ICP final error {err:.2f} m over {travelled[0]:.0f} m"
    )
