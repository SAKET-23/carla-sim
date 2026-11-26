"""launch_testing: gt-backend segmentation, detection and tracker nodes on synthetic data (ROS dev container only)."""

import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.markers
import numpy as np
import pytest
import rclpy
from hybrid_av_msgs.msg import Track, TrackArray
from hybrid_av_nodes import convert as C
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float32
from vision_msgs.msg import Detection3DArray

from hybrid_av.types import EgoState, ObjectClass, ObjectState

pytestmark = pytest.mark.ros
RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
TICKS = 4


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    nodes = [
        launch_ros.actions.Node(package="hybrid_av_nodes", executable=e, parameters=[p], output="screen")
        for e, p in (
            ("segmentation_node", {"backend": "gt"}),
            ("detection_node", {"backend": "gt"}),
            ("tracker_node", {"mode": "ground_truth"}),
        )
    ]
    return launch.LaunchDescription([*nodes, launch_testing.actions.ReadyToTest()])


def stamp_of(m):
    return C.stamp_to_sec(m.header.stamp)


def cloud(names, arr, nanosec):
    m = PointCloud2()
    C.fill_pointcloud2(m, np.asarray(arr, np.float32).reshape(-1, len(names)), names, PointField)
    m.header.stamp.sec, m.header.stamp.nanosec = 1, nanosec
    return m


class TestPerceptionNodes(unittest.TestCase):
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

    def test_gt_pipeline(self):
        n = rclpy.create_node("perception_tester")
        seg, ratio, dets, tracks = [], [], [], []
        n.create_subscription(Image, "/perception/segmentation", seg.append, qos_profile_sensor_data)
        n.create_subscription(Float32, "/perception/drivable_ratio", ratio.append, RELIABLE)
        n.create_subscription(Detection3DArray, "/perception/detections3d", dets.append, RELIABLE)
        n.create_subscription(TrackArray, "/perception/tracks3d", tracks.append, RELIABLE)
        pubs = {
            "img": n.create_publisher(
                Image, "/carla/ego/camera/front_semantic/image", qos_profile_sensor_data
            ),
            "lidar": n.create_publisher(PointCloud2, "/carla/ego/lidar/points", qos_profile_sensor_data),
            "objects": n.create_publisher(TrackArray, "/carla/ground_truth/objects", RELIABLE),
            "odom": n.create_publisher(Odometry, "/carla/ego/ground_truth/odom", RELIABLE),
            **{
                s: n.create_publisher(PointCloud2, f"/carla/ego/radar/{s}", qos_profile_sensor_data)
                for s in ("front", "rear", "left", "right")
            },
        }
        want = {"img": 1, "lidar": 1, "objects": 1, "odom": 2, "front": 1, "rear": 1, "left": 1, "right": 1}
        self.spin_until(n, lambda: all(pubs[k].get_subscription_count() >= v for k, v in want.items()))
        time.sleep(1.5)  # the readers learn of our writers slightly later
        radar_names = ("x", "y", "z", "velocity", "azimuth", "altitude", "depth")
        for k in range(TICKS):
            ns, cx = k * 50_000_000, 10.0 + 0.5 * k
            img = Image()
            img.height = img.width = 8
            img.encoding, img.step = "mono8", 8
            img.data = bytes([1] * 64)  # CARLA tag 1 = Roads -> train id 0
            img.header.stamp.sec, img.header.stamp.nanosec = 1, ns
            pubs["img"].publish(img)
            xs = np.linspace(cx - 1.5, cx + 1.5, 20)
            pts = np.column_stack([xs, np.linspace(-0.5, 0.5, 20), np.full(20, 0.75), np.ones(20)])
            pubs["lidar"].publish(cloud(("x", "y", "z", "intensity"), pts, ns))
            car = ObjectState(id=1, cls=ObjectClass.CAR, x=cx, y=0.0, z=0.75, yaw=0.0, length=4.5, width=2.0,
                              height=1.5, vx=10.0, vy=0.0, pos_cov=None)  # fmt: skip
            objs = TrackArray()
            C.fill_track_array(objs, [car], Track)
            objs.header.stamp.sec, objs.header.stamp.nanosec = 1, ns
            pubs["objects"].publish(objs)
            odom = Odometry()
            C.fill_odometry(
                odom, EgoState(t=0.0, x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0), "map", "base_link"
            )
            odom.header.stamp.sec, odom.header.stamp.nanosec = 1, ns
            pubs["odom"].publish(odom)
            for s in ("front", "rear", "left", "right"):
                pubs[s].publish(cloud(radar_names, np.zeros((0, 7)), ns))
            self.spin_until(n, lambda k=k: len(dets) > k and len(tracks) > k)
        self.spin_until(n, lambda: len(seg) >= 1 and len(ratio) >= 1)
        n.destroy_node()
        assert stamp_of(seg[0]) == 1.0 and not np.frombuffer(seg[0].data, np.uint8).any()
        assert ratio[0].data == 1.0
        assert stamp_of(dets[0]) == 1.0 and len(dets[0].detections) == 1
        assert dets[0].detections[0].results[0].hypothesis.class_id == "CAR"
        assert dets[0].header.frame_id == "base_link"
        last = tracks[-1]
        assert last.header.frame_id == "map" and len(last.tracks) == 1 and last.tracks[0].velocity.x > 0
