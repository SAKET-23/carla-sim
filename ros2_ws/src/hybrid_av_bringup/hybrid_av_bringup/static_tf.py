"""static_tf: publishes /tf_static (base_link -> sensor frames) from the sensor rig yaml."""

import rclpy
from geometry_msgs.msg import TransformStamped
from hybrid_av_nodes.rig import load_rig
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from hybrid_av_bringup.tf_math import quat_from_rpy


class StaticTf(Node):
    def __init__(self) -> None:
        super().__init__("static_tf")
        rig = load_rig(self.declare_parameter("rig", "").value)
        out = []
        for parent, child, (x, y, z, roll, pitch, yaw) in rig.static_transforms:
            m = TransformStamped()  # stamp 0: static transforms are valid for all time
            m.header.frame_id, m.child_frame_id = parent, child
            m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = x, y, z
            q = m.transform.rotation
            q.x, q.y, q.z, q.w = quat_from_rpy(roll, pitch, yaw)
            out.append(m)
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(out)


def main() -> None:
    rclpy.init()
    node = StaticTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
