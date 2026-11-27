"""launch_testing: litesim_node + policy_node + mpc_node closed loop (ROS dev container only)."""

import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from hybrid_av_msgs.msg import FeatureVector, VehicleCommand
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

pytestmark = pytest.mark.ros
TICKS = 40


def _node(executable, sim_time=True, **params):
    return launch_ros.actions.Node(
        package="hybrid_av_nodes", executable=executable, parameters=[{"use_sim_time": sim_time, **params}],
        output="screen",
    )  # fmt: skip


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    return launch.LaunchDescription(
        [
            _node("policy_node", policy="constant"),
            _node("mpc_node", mode="ground_truth"),
            _node("litesim_node", sim_time=False, scenario="nominal", seed=0, max_ticks=TICKS),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec / 1e9


class TestMpcNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_one_command_per_tick(self):
        n = rclpy.create_node("mpc_tester")
        clocks, cmds, feats = [], [], []
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=100)
        n.create_subscription(
            Clock, "/clock", lambda m: clocks.append(m.clock.sec + m.clock.nanosec / 1e9), qos
        )
        n.create_subscription(VehicleCommand, "/carla/ego/command", lambda m: cmds.append(stamp(m)), qos)
        n.create_subscription(FeatureVector, "/autonomy/features", feats.append, qos)
        end = time.monotonic() + 120
        # /clock and command are separate subscriptions; spin_once serves one per call, so wait for both counts
        while time.monotonic() < end and (len(cmds) < TICKS or len(clocks) < TICKS):
            rclpy.spin_once(n, timeout_sec=0.05)
        n.destroy_node()
        assert len(clocks) == TICKS and len(cmds) == TICKS
        assert all(abs(c - t) < 1e-6 for c, t in zip(cmds, clocks))
        assert feats and all(f.source == "ground_truth" for f in feats)
