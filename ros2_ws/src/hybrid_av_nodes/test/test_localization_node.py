"""launch_testing: localization_node in both modes (ROS dev container only). The perception node's TF is
remapped to /tf_perception so the two modes' map -> odom edges can be told apart."""

import math
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage

pytestmark = pytest.mark.ros
RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    gt = launch_ros.actions.Node(
        package="hybrid_av_nodes", executable="localization_node", name="loc_gt",
        parameters=[{"mode": "ground_truth"}], output="screen",
    )  # fmt: skip
    perc = launch_ros.actions.Node(
        package="hybrid_av_nodes", executable="localization_node", name="loc_perc",
        parameters=[{"mode": "perception"}], remappings=[("/tf", "/tf_perception")], output="screen",
    )  # fmt: skip
    return launch.LaunchDescription([gt, perc, launch_testing.actions.ReadyToTest()])


def odom(sec, x, y, yaw):
    m = Odometry()
    m.header.stamp.sec = sec
    m.pose.pose.position.x, m.pose.pose.position.y = x, y
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return m


class TestLocalizationNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def spin_until(self, n, cond, timeout=30.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not cond():
            rclpy.spin_once(n, timeout_sec=0.05)
        assert cond()

    def test_both_modes(self):
        n = rclpy.create_node("loc_tester")
        tf, tf_p, ego = [], [], []
        n.create_subscription(TFMessage, "/tf", lambda m: tf.extend(m.transforms), 10)
        n.create_subscription(TFMessage, "/tf_perception", lambda m: tf_p.extend(m.transforms), 10)
        n.create_subscription(Odometry, "/localization/ego_state", ego.append, RELIABLE)
        gt_pub = n.create_publisher(Odometry, "/carla/ego/ground_truth/odom", RELIABLE)
        kiss_pub = n.create_publisher(Odometry, "/kiss/odometry", 10)
        wheel_pub = n.create_publisher(Odometry, "/carla/ego/wheel_odom", 10)
        imu_pub = n.create_publisher(Imu, "/carla/ego/imu", 10)
        self.spin_until(
            n,
            lambda: gt_pub.get_subscription_count() >= 2 and kiss_pub.get_subscription_count() >= 1
            and wheel_pub.get_subscription_count() >= 1 and imu_pub.get_subscription_count() >= 1,
        )  # fmt: skip
        time.sleep(1.0)  # the readers learn of our writers slightly later
        gt_pub.publish(odom(1, 100.0, 50.0, math.pi / 2))
        wheel = odom(1, 0.0, 0.0, 0.0)
        wheel.twist.twist.linear.x = 10.0
        wheel_pub.publish(wheel)
        imu = Imu()
        imu.angular_velocity.z = 0.02
        imu_pub.publish(imu)
        self.spin_until(n, lambda: any(t.child_frame_id == "base_link" for t in tf))
        kiss_pub.publish(odom(1, 0.0, 0.0, 0.0))
        self.spin_until(n, lambda: len(ego) >= 1)
        kiss_pub.publish(odom(2, 1.0, 0.0, 0.0))
        self.spin_until(n, lambda: len(ego) >= 2)
        n.destroy_node()
        # ground_truth mode: identity map -> odom, GT pose as odom -> base_link
        m2o = next(t for t in tf if t.child_frame_id == "odom")
        assert m2o.transform.translation.x == 0.0 and m2o.transform.rotation.w == 1.0
        o2b = next(t for t in tf if t.child_frame_id == "base_link")
        assert (o2b.transform.translation.x, o2b.transform.translation.y) == (100.0, 50.0)
        # perception mode: aligned ego state in map, constant map -> odom
        e = ego[-1]
        assert e.header.stamp.sec == 2 and e.header.frame_id == "map" and e.child_frame_id == "base_link"
        assert (e.pose.pose.position.x, e.pose.pose.position.y) == pytest.approx((100.0, 51.0))
        assert (e.twist.twist.linear.x, e.twist.twist.angular.z) == pytest.approx((10.0, 0.02))
        a = tf_p[-1]
        assert a.header.frame_id == "map" and a.child_frame_id == "odom"
        assert (a.transform.translation.x, a.transform.translation.y) == pytest.approx((100.0, 50.0))
