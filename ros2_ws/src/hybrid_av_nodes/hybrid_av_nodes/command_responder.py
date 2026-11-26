"""command_responder: answers every /clock with a full-brake command carrying that stamp.

Stand-in for mpc_node in bridge_only.launch.py, so the bridge never waits for its command deadline."""

import rclpy
from hybrid_av_msgs.msg import VehicleCommand
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


class CommandResponder(Node):
    def __init__(self) -> None:
        super().__init__("command_responder")
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
        self.pub = self.create_publisher(VehicleCommand, "/carla/ego/command", reliable)
        self.create_subscription(Clock, "/clock", self._on_clock, reliable)

    def _on_clock(self, m: Clock) -> None:
        cmd = VehicleCommand()
        cmd.header.stamp = m.clock
        cmd.steer, cmd.throttle, cmd.brake = 0.0, 0.0, 1.0
        self.pub.publish(cmd)


def main() -> None:
    rclpy.init()
    node = CommandResponder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
