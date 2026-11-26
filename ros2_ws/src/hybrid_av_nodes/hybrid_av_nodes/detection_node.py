"""detection_node: lidar points -> /perception/detections3d in base_link. The gt backend needs GT objects and ego."""

from pathlib import Path

import rclpy
import yaml
from hybrid_av_msgs.msg import TrackArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from hybrid_av.perception.base import make_detector
from hybrid_av_nodes import convert as C
from hybrid_av_nodes.rig import lidar_to_base, load_rig
from hybrid_av_nodes.stamp_sync import StampSync


def _detection() -> Detection3D:
    d = Detection3D()
    d.results = [ObjectHypothesisWithPose()]
    return d


class DetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("detection_node")
        p = self.declare_parameter
        self.backend = p("backend", "gt").value
        config, rig = p("config", "").value, p("rig", "").value
        cfg = yaml.safe_load(Path(config).read_text()) if config else {}
        self.detector = make_detector({**cfg, "backend": self.backend})
        # the lidar frame is only transformed when a rig is given (interfaces.md section 14)
        self.mount = (
            next(t for _, c, t in load_rig(rig).static_transforms if c == "lidar_top") if rig else None
        )
        self.unwrap = C.HeadingUnwrapper()
        self.sync = StampSync(("cloud", "objects", "odom") if self.backend == "gt" else ("cloud",))
        reliable = lambda depth: QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=depth)
        self.pub = self.create_publisher(Detection3DArray, "/perception/detections3d", reliable(5))
        self.create_subscription(
            PointCloud2, "/carla/ego/lidar/points", lambda m: self._add("cloud", m), qos_profile_sensor_data
        )
        if self.backend == "gt":
            self.create_subscription(
                TrackArray, "/carla/ground_truth/objects", lambda m: self._add("objects", m), reliable(10)
            )
            self.create_subscription(
                Odometry, "/carla/ego/ground_truth/odom", lambda m: self._add("odom", m), reliable(10)
            )

    def _add(self, key: str, m) -> None:
        s = self.sync.add(key, C.stamp_to_sec(m.header.stamp), m)
        if s is None:
            return
        cloud = s["cloud"]
        points = C.xyzi_from_pointcloud2(cloud)
        if self.mount is not None:
            points = lidar_to_base(points, self.mount)
        if self.backend == "gt":
            self.detector.set_ground_truth(
                C.objects_from_track_array(s["objects"]), C.ego_from_odometry(s["odom"], self.unwrap)
            )
        out = Detection3DArray()
        out.header.stamp, out.header.frame_id = cloud.header.stamp, "base_link"
        C.fill_detection_array(out, self.detector(points), _detection)
        for d in out.detections:
            d.header = out.header
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
