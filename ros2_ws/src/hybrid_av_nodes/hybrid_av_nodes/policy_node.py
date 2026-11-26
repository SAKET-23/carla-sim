"""policy_node: FeatureVector in, PolicyAction out on policy ticks; LoadPolicy hot swap."""

import rclpy
from hybrid_av_msgs.msg import FeatureVector, PolicyAction
from hybrid_av_msgs.srv import LoadPolicy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from hybrid_av_nodes import convert as C
from hybrid_av_nodes.policy_core import PolicyCore, PolicyTickGate


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")
        self.core = PolicyCore(self.declare_parameter("policy", "constant").value)
        self.gate = PolicyTickGate(
            float(self.declare_parameter("dt", 0.05).value),
            int(self.declare_parameter("policy_period", 2).value),
        )
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
        self.pub = self.create_publisher(PolicyAction, "/autonomy/policy_action", qos)
        self.create_subscription(FeatureVector, "/autonomy/features", self._on_features, qos)
        self.create_service(LoadPolicy, "/autonomy/load_policy", self._on_load)

    def _on_features(self, m: FeatureVector) -> None:
        if not self.gate(C.stamp_to_sec(m.header.stamp)):
            return
        action, h = self.core.act(C.frame_from_feature_msg(m))
        out = PolicyAction()
        out.header.stamp = m.header.stamp
        C.fill_action(out, action, h)
        self.pub.publish(out)

    def _on_load(self, req: LoadPolicy.Request, res: LoadPolicy.Response) -> LoadPolicy.Response:
        res.success, res.policy_hash, res.message = self.core.load(req.path)
        (self.get_logger().info if res.success else self.get_logger().error)(res.message)
        return res


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
