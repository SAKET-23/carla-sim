"""launch_testing: policy_node answers a FeatureVector and hot-swaps via LoadPolicy (ROS dev container only)."""

import tempfile
import time
import unittest
from pathlib import Path

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
import torch
from hybrid_av_msgs.msg import FeatureVector, PolicyAction
from hybrid_av_msgs.srv import LoadPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy

from hybrid_av.policy.io import save_policy
from hybrid_av.policy.network import CompactPolicy

pytestmark = pytest.mark.ros
QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    node = launch_ros.actions.Node(
        package="hybrid_av_nodes",
        executable="policy_node",
        parameters=[{"policy": "constant"}],
        output="screen",
    )
    return launch.LaunchDescription([node, launch_testing.actions.ReadyToTest()])


class TestPolicyNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def _action_for(self, n, pub, got, sec):
        m = FeatureVector()
        m.header.stamp.sec = sec
        m.normalized = [0.0] * 24
        m.raw = [0.0] * 24
        end = time.monotonic() + 15
        while time.monotonic() < end and not got:
            pub.publish(m)  # re-sent until the discovery handshake is done; the node answers each one
            rclpy.spin_once(n, timeout_sec=0.1)
        return got[-1]

    def test_action_and_hot_swap(self):
        n = rclpy.create_node("policy_tester")
        got = []
        pub = n.create_publisher(FeatureVector, "/autonomy/features", QOS)
        n.create_subscription(PolicyAction, "/autonomy/policy_action", got.append, QOS)
        first = self._action_for(n, pub, got, 5)
        assert first.header.stamp.sec == 5 and first.policy_hash == "constant"
        torch.manual_seed(0)
        path = Path(tempfile.mkdtemp()) / "p.pt"
        save_policy(CompactPolicy(), path, {})
        cli = n.create_client(LoadPolicy, "/autonomy/load_policy")
        assert cli.wait_for_service(timeout_sec=10)
        req = LoadPolicy.Request()
        req.path = str(path)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(n, fut, timeout_sec=10)
        assert fut.result().success and fut.result().policy_hash != "constant"
        got.clear()
        again = self._action_for(n, pub, got, 5)
        assert again.policy_hash == fut.result().policy_hash
        n.destroy_node()
