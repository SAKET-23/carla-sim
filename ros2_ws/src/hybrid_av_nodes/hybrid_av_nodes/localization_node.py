"""localization_node: map -> odom (and odom -> base_link in ground_truth mode); /localization/ego_state in perception mode."""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from hybrid_av_nodes import convert as C
from hybrid_av_nodes.localization_core import LocalizationCore


def _tf(stamp, parent, child, x, y, yaw) -> TransformStamped:
    m = TransformStamped()
    m.header.stamp, m.header.frame_id, m.child_frame_id = stamp, parent, child
    m.transform.translation.x, m.transform.translation.y = x, y
    C.fill_quat_from_yaw(m.transform.rotation, yaw)
    return m


class LocalizationNode(Node):
    def __init__(self) -> None:
        super().__init__("localization_node")
        self.mode = self.declare_parameter("mode", "ground_truth").value
        if self.mode not in ("ground_truth", "perception"):
            raise ValueError(f"mode must be ground_truth or perception, got {self.mode!r}")
        self.core = LocalizationCore(float(self.declare_parameter("divergence_m", 2.0).value))
        self.tf = TransformBroadcaster(self)
        reliable = lambda depth: QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=depth)
        self.create_subscription(Odometry, "/carla/ego/ground_truth/odom", self._on_gt, reliable(10))
        if self.mode == "perception":
            self.pub = self.create_publisher(Odometry, "/localization/ego_state", reliable(10))
            self.create_subscription(Odometry, "/kiss/odometry", self._on_kiss, 10)
            self.create_subscription(
                Odometry, "/carla/ego/wheel_odom", self._on_wheel, qos_profile_sensor_data
            )
            self.create_subscription(Imu, "/carla/ego/imu", self._on_imu, qos_profile_sensor_data)

    def _on_gt(self, m: Odometry) -> None:
        ego = C.ego_from_odometry(m, C.HeadingUnwrapper())
        self.core.on_ground_truth(ego.t, ego)
        if self.mode == "ground_truth":  # KISS-ICP is not launched: the GT pose is odom -> base_link (IC-31)
            s = m.header.stamp
            self.tf.sendTransform(
                [_tf(s, "map", "odom", 0.0, 0.0, 0.0), _tf(s, "odom", "base_link", ego.x, ego.y, ego.psi)]
            )

    def _on_wheel(self, m: Odometry) -> None:
        self.core.on_wheel(C.stamp_to_sec(m.header.stamp), m.twist.twist.linear.x, m.twist.twist.linear.y)

    def _on_imu(self, m: Imu) -> None:
        self.core.on_imu(C.stamp_to_sec(m.header.stamp), m.angular_velocity.z)

    def _on_kiss(self, m: Odometry) -> None:
        was_failed, t = self.core.localization_failed, C.stamp_to_sec(m.header.stamp)
        p = m.pose.pose.position
        ego = self.core.on_odom(t, p.x, p.y, C.yaw_from_quat(m.pose.pose.orientation))
        if ego is None:
            return
        out = Odometry()
        out.header.stamp = m.header.stamp
        C.fill_odometry(out, ego, "map", "base_link")
        self.pub.publish(out)
        self.tf.sendTransform(_tf(m.header.stamp, "map", "odom", *self.core.map_to_odom()))
        if self.core.localization_failed and not was_failed:
            self.get_logger().error(
                "localization diverged from ground truth (reported only, not used for control)"
            )


def main() -> None:
    rclpy.init()
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
