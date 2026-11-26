"""Smoke 05 (runbook.md section 4): message counts of the bridge topics per /clock tick over --seconds of wall time.

Pass: LiDAR, camera and GT odometry within +-2 % of the /clock count; IMU within +-2 % of 1x or 5x the /clock count
(IC-10); /clock count / seconds >= 5 Hz. Prints the wall-clock tick rate (cost-and-hours.md section 1).
"""

import argparse
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, Imu, PointCloud2

TOPICS = {
    "/clock": Clock,
    "/carla/ego/lidar/points": PointCloud2,
    "/carla/ego/camera/front/image": Image,
    "/carla/ego/imu": Imu,
    "/carla/ego/ground_truth/odom": Odometry,
}


def judge(counts: dict[str, int], seconds: float) -> list[str]:
    """Failure messages (empty = pass)."""
    clock, bad = counts["/clock"], []
    if clock / seconds < 5.0:
        bad.append(f"/clock rate {clock / seconds:.2f} Hz < 5 Hz")
    for t in ("/carla/ego/lidar/points", "/carla/ego/camera/front/image", "/carla/ego/ground_truth/odom"):
        if abs(counts[t] - clock) > 0.02 * clock:
            bad.append(f"{t}: {counts[t]} messages vs {clock} ticks")
    imu = counts["/carla/ego/imu"]
    if not any(abs(imu - k * clock) <= 0.02 * k * clock for k in (1, 5)):
        bad.append(f"/carla/ego/imu: {imu} messages vs {clock} ticks (expected 1x or 5x)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    rclpy.init()
    node = Node("bridge_rate_check")
    counts = dict.fromkeys(TOPICS, 0)
    # best effort with a deep queue matches both reliable and sensor-data publishers (interfaces.md section 14)
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1000)
    for topic, typ in TOPICS.items():
        node.create_subscription(typ, topic, lambda _m, t=topic: counts.__setitem__(t, counts[t] + 1), qos)
    end = time.monotonic() + a.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    print("counts:", counts)
    print(f"wall-clock tick rate: {counts['/clock'] / a.seconds:.2f} Hz")
    bad = judge(counts, a.seconds)
    for b in bad:
        print("FAIL:", b, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
