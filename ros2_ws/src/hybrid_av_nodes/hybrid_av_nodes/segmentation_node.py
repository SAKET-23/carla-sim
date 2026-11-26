"""segmentation_node: camera (or CARLA semantic camera for backend gt) -> train-id image and drivable ratio."""

from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from hybrid_av.perception.base import make_segmenter
from hybrid_av.perception.classes import drivable_ratio


class SegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__("segmentation_node")
        backend = self.declare_parameter("backend", "gt").value
        config = self.declare_parameter("config", "").value
        cfg = yaml.safe_load(Path(config).read_text()) if config else {}
        self.segmenter = make_segmenter({**cfg, "backend": backend})
        topic = (
            "/carla/ego/camera/front_semantic/image" if backend == "gt" else "/carla/ego/camera/front/image"
        )
        self.pub = self.create_publisher(Image, "/perception/segmentation", qos_profile_sensor_data)
        self.ratio_pub = self.create_publisher(
            Float32, "/perception/drivable_ratio", QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
        )
        self.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)

    def _on_image(self, m: Image) -> None:
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        img = img[..., 0] if m.encoding == "mono8" else img  # mono8 arrives as (H, W, 1)
        ids = np.ascontiguousarray(self.segmenter(img), dtype=np.uint8)
        out = Image()
        out.header = m.header
        out.height, out.width, out.encoding, out.step = ids.shape[0], ids.shape[1], "mono8", ids.shape[1]
        out.data = ids.tobytes()
        self.pub.publish(out)
        self.ratio_pub.publish(Float32(data=drivable_ratio(ids)))


def main() -> None:
    rclpy.init()
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
