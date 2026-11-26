"""launch_testing: litesim_node with a test-local command responder (runs in the ROS dev container only)."""

import itertools
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from hybrid_av_msgs.msg import RouteInfo, VehicleCommand
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

pytestmark = pytest.mark.ros


@pytest.mark.launch_test
@launch_testing.markers.keep_alive  # the node exits at episode end; the test still has to finish
def generate_test_description():
    node = launch_ros.actions.Node(
        package="hybrid_av_nodes",
        executable="litesim_node",
        parameters=[{"scenario": "nominal", "seed": 0, "max_ticks": 20}],
        output="screen",
    )
    return launch.LaunchDescription([node, launch_testing.actions.ReadyToTest()]), {"node": node}


class TestLiteSimNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_clock_route_and_stamps(self):
        n = rclpy.create_node("responder")
        clocks, odoms, routes = [], [], []
        cmd_pub = n.create_publisher(VehicleCommand, "/carla/ego/command", 1)

        def on_clock(m):
            clocks.append(m.clock.sec + m.clock.nanosec / 1e9)
            cmd = VehicleCommand()
            cmd.header.stamp = m.clock
            cmd.brake = 1.0
            cmd_pub.publish(cmd)

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
        n.create_subscription(Clock, "/clock", on_clock, qos)
        n.create_subscription(Odometry, "/carla/ego/ground_truth/odom", lambda m: odoms.append(m), 10)
        late = None
        end = time.monotonic() + 60
        while time.monotonic() < end and len(clocks) < 20:
            rclpy.spin_once(n, timeout_sec=0.05)
            if late is None and len(clocks) >= 5:  # a subscriber that joins mid-episode still gets the route
                late = n.create_subscription(
                    RouteInfo, "/carla/ego/route", routes.append,
                    QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1),
                )  # fmt: skip
        for _ in range(20):
            rclpy.spin_once(n, timeout_sec=0.05)
        n.destroy_node()
        assert len(clocks) == 20
        assert all(abs(b - a - 0.05) < 1e-9 for a, b in itertools.pairwise(clocks))
        assert len(routes) == 1 and len(routes[0].x) > 10
        stamps = {o.header.stamp.sec + o.header.stamp.nanosec / 1e9 for o in odoms}
        assert odoms and stamps <= set(clocks)


@launch_testing.post_shutdown_test()
class TestExit(unittest.TestCase):
    def test_exit_code(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
