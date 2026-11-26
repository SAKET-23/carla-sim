"""tracker_node: detections + radar + ego -> /perception/tracks3d in the map frame."""

import rclpy
from hybrid_av_msgs.msg import Track, TrackArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray

from hybrid_av.config import load_dataclass
from hybrid_av.tracking.tracker import Tracker, TrackerConfig
from hybrid_av_nodes import convert as C
from hybrid_av_nodes.rig import load_rig
from hybrid_av_nodes.stamp_sync import StampSync

_SIDES = ("front", "rear", "left", "right")


class TrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("tracker_node")
        p = self.declare_parameter
        mode, rig, config = p("mode", "ground_truth").value, p("rig", "").value, p("config", "").value
        if mode not in ("ground_truth", "perception"):
            raise ValueError(f"mode must be ground_truth or perception, got {mode!r}")
        radars = load_rig(rig).radars if rig else []
        self.tracker = Tracker(load_dataclass(TrackerConfig, config) if config else TrackerConfig(), radars)
        self.unwrap = C.HeadingUnwrapper()
        self.sync = StampSync(("det", "ego", *(f"radar_{s}" for s in _SIDES)))
        reliable = lambda depth: QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=depth)
        self.pub = self.create_publisher(TrackArray, "/perception/tracks3d", reliable(10))
        self.create_subscription(
            Detection3DArray, "/perception/detections3d", lambda m: self._add("det", m), reliable(5)
        )
        ego_topic = "/localization/ego_state" if mode == "perception" else "/carla/ego/ground_truth/odom"
        self.create_subscription(Odometry, ego_topic, lambda m: self._add("ego", m), reliable(10))
        for side in _SIDES:
            self.create_subscription(
                PointCloud2, f"/carla/ego/radar/{side}",
                lambda m, k=f"radar_{side}": self._add(k, m), qos_profile_sensor_data,
            )  # fmt: skip

    def _add(self, key: str, m) -> None:
        t = C.stamp_to_sec(m.header.stamp)
        s = self.sync.add(key, t, m)
        if s is None:
            return
        ego = C.ego_from_odometry(s["ego"], self.unwrap)
        radar = [
            d
            for side in _SIDES
            if s[f"radar_{side}"].width  # an empty cloud carries no fields to read
            for d in C.radar_from_pointcloud2(s[f"radar_{side}"], side)
        ]
        self.tracker.predict(t)
        self.tracker.update_boxes(t, C.boxes_from_detection_array(s["det"]), ego)
        self.tracker.update_radar(t, radar, ego)
        out = TrackArray()
        out.header.stamp, out.header.frame_id = s["det"].header.stamp, "map"
        C.fill_track_array(out, self.tracker.tracks(), Track)
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
