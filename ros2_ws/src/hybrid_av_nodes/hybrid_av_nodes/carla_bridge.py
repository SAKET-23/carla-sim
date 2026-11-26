"""carla_bridge: owns CarlaSim and the tick, publishes the CARLA topics of interfaces.md section 14.

Mirrors litesim_node: tick 0 publishes the reset observation without stepping, `strict_peers` waits for the
whole stack before tick 0. `carla` is only imported inside CarlaSim."""

import json
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from hybrid_av_msgs.msg import RouteInfo, Track, TrackArray, TrafficContext, VehicleCommand
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
from tf2_ros import TransformBroadcaster

from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.carla_backend import convert as CC
from hybrid_av.sim.carla_backend.client import CarlaSimConfig
from hybrid_av.sim.carla_backend.routes import carla_scenario
from hybrid_av.sim.carla_backend.sensors import load_rig as load_specs
from hybrid_av.sim.carla_backend.sim import CarlaSim
from hybrid_av_nodes import bridge_core as B
from hybrid_av_nodes import convert as C
from hybrid_av_nodes.rig import load_rig
from hybrid_av_nodes.tick_loop import CommandMailbox, TickLoop

PEER_SETTLE_S = 0.5
PEER_WAIT_S = 30.0
_SIDES = ("front", "rear", "left", "right")
_RADAR_FIELDS = ("x", "y", "z", "velocity", "azimuth", "altitude", "depth")


class CarlaBridge(Node):
    def __init__(self) -> None:
        super().__init__("carla_bridge")
        p = self.declare_parameter
        self.town = p("town", "").value
        self.scenario_name = p("scenario", "nominal").value
        self.seed = int(p("seed", 0).value)
        self.quality = p("quality", "Low").value
        self.sensors = p("sensors", "").value  # rig yaml; empty = no sensors
        self.routes = p("routes", "").value  # carla_routes.yaml; empty = next to the rig's configs/
        self.command_timeout = float(p("command_timeout", 1.0).value)
        self.warmup_ticks = int(p("warmup_ticks", 20).value)
        self.max_ticks = int(p("max_ticks", 0).value)
        self.result_path = p("result_path", "").value
        self.no_rendering = bool(p("no_rendering", False).value)
        self.strict_peers = bool(p("strict_peers", False).value)
        self.wheel = B.WheelOdomSynth(int(p("wheel_odom_seed", 0).value))

        reliable = lambda depth: QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=depth)
        sensor = qos_profile_sensor_data
        mk = self.create_publisher
        self.clock_pub = mk(Clock, "/clock", reliable(1))
        self.odom_pub = mk(Odometry, "/carla/ego/ground_truth/odom", reliable(10))
        self.objects_pub = mk(TrackArray, "/carla/ground_truth/objects", reliable(10))
        self.context_pub = mk(TrafficContext, "/carla/ego/traffic_context", reliable(10))
        self.wheel_pub = mk(Odometry, "/carla/ego/wheel_odom", sensor)
        self.image_pub = mk(Image, "/carla/ego/camera/front/image", sensor)
        self.info_pub = mk(CameraInfo, "/carla/ego/camera/front/camera_info", sensor)
        self.sem_pub = mk(Image, "/carla/ego/camera/front_semantic/image", sensor)
        self.lidar_pub = mk(PointCloud2, "/carla/ego/lidar/points", sensor)
        self.radar_pubs = {s: mk(PointCloud2, f"/carla/ego/radar/{s}", sensor) for s in _SIDES}
        self.imu_pub = mk(Imu, "/carla/ego/imu", sensor)
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1
        )
        self.route_pub = mk(RouteInfo, "/carla/ego/route", latched)
        self.tf = TransformBroadcaster(self)
        self.mailbox = CommandMailbox()
        self.create_subscription(VehicleCommand, "/carla/ego/command", self._on_command, reliable(1))

        self.radars = {r.sensor_id: r for r in load_rig(self.sensors).radars} if self.sensors else {}
        cam = (
            next((s for s in load_specs(self.sensors) if s.id == "camera_front"), None)
            if self.sensors
            else None
        )
        self.cam_info = (
            B.camera_info(
                int(cam.attributes["image_size_x"]),
                int(cam.attributes["image_size_y"]),
                float(cam.attributes["fov"]),
            )
            if cam
            else None
        )
        self.skipped: dict[str, int] = {}
        self.driver = None
        self.stats = None
        self.failed = False
        self.done = threading.Event()

    def _on_command(self, m: VehicleCommand) -> None:
        self.mailbox.put(C.stamp_to_sec(m.header.stamp), C.command_from_msg(m))

    # --- publishing -----------------------------------------------------------------
    def publish(self, t: float) -> None:
        sim, obs = self.driver.sim, self.driver.obs
        ego, stamp = obs.ego, Clock()
        C.fill_stamp(stamp.clock, t)
        self.clock_pub.publish(stamp)  # first within the tick (ECC-20)

        def stamped(msg, frame_id=""):
            msg.header.stamp, msg.header.frame_id = stamp.clock, frame_id
            return msg

        odom = stamped(Odometry())
        C.fill_odometry(odom, ego, "map", "base_link")
        self.odom_pub.publish(odom)
        objs = stamped(TrackArray(), "map")
        C.fill_track_array(objs, obs.objects, Track)
        self.objects_pub.publish(objs)
        ctx = stamped(TrafficContext(), "map")
        C.fill_context(ctx, obs.context)
        self.context_pub.publish(ctx)
        wheel = stamped(Odometry(), "base_link")
        wheel.child_frame_id = "base_link"
        wheel.twist.twist.linear.x, wheel.twist.twist.linear.y = self.wheel(ego.vx, ego.vy)
        wheel.twist.twist.angular.z = ego.r
        self.wheel_pub.publish(wheel)
        tf = stamped(TransformStamped(), "map")
        tf.child_frame_id = "ground_truth_base_link"
        tf.transform.translation.x, tf.transform.translation.y = ego.x, ego.y
        C.fill_quat_from_yaw(tf.transform.rotation, ego.psi)
        self.tf.sendTransform(tf)
        self._publish_sensors(sim.last_sensor_data, stamp.clock, t)

    def _publish_sensors(self, data: dict, stamp, t: float) -> None:
        for sid, datum in data.items():
            if datum is None:  # a missing sensor is skipped for this tick and counted
                self.skipped[sid] = self.skipped.get(sid, 0) + 1
        if (d := data.get("camera_front")) is not None:
            rgb = B.image_bgra_to_rgb(d.raw_data, d.width, d.height)
            self.image_pub.publish(self._image(rgb, "rgb8", stamp))
            self.info_pub.publish(self._info(stamp))
        if (d := data.get("camera_front_semantic")) is not None:
            self.sem_pub.publish(self._image(B.semantic_tags(d.raw_data, d.width, d.height), "mono8", stamp))
        if (d := data.get("lidar")) is not None:
            pts = CC.lidar_to_lib(np.frombuffer(d.raw_data, np.float32))
            msg = PointCloud2()
            msg.header.stamp, msg.header.frame_id = stamp, "lidar_top"
            C.fill_pointcloud2(msg, pts, ("x", "y", "z", "intensity"), PointField)
            self.lidar_pub.publish(msg)
        for side, pub in self.radar_pubs.items():
            if (d := data.get(f"radar_{side}")) is not None and side in self.radars:
                dets = CC.radar_to_lib(np.frombuffer(d.raw_data, np.float32), side)
                msg = PointCloud2()
                msg.header.stamp, msg.header.frame_id = stamp, "base_link"  # x, y, z are in base_link
                C.fill_pointcloud2(msg, B.radar_points(dets, self.radars[side]), _RADAR_FIELDS, PointField)
                pub.publish(msg)
        if samples := data.get("imu"):  # ECC-07: one message per sample, stamps interpolated
            for sample, ts in zip(samples, B.imu_stamps(t, len(samples))):
                a, g = CC.imu_to_lib(sample.accelerometer, sample.gyroscope)
                msg = Imu()
                C.fill_stamp(msg.header.stamp, ts)
                msg.header.frame_id = "imu"
                msg.orientation_covariance[0] = -1.0  # no orientation estimate
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(
                    float, a
                )
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, g)
                self.imu_pub.publish(msg)

    @staticmethod
    def _image(arr: np.ndarray, encoding: str, stamp) -> Image:
        msg = Image()
        msg.header.stamp, msg.header.frame_id = stamp, "camera_front_optical"
        msg.height, msg.width, msg.encoding = arr.shape[0], arr.shape[1], encoding
        msg.is_bigendian, msg.step, msg.data = (
            0,
            arr.shape[1] * (arr.shape[2] if arr.ndim == 3 else 1),
            arr.tobytes(),
        )
        return msg

    def _info(self, stamp) -> CameraInfo:
        ci, msg = self.cam_info, CameraInfo()
        msg.header.stamp, msg.header.frame_id = stamp, "camera_front_optical"
        msg.width, msg.height, msg.distortion_model = ci["width"], ci["height"], ci["distortion_model"]
        msg.d, msg.k, msg.r, msg.p = ci["d"], ci["k"], ci["r"], ci["p"]
        return msg

    # --- run ------------------------------------------------------------------------------
    def _peers_ready(self) -> bool:
        ok = bool(self.clock_pub.get_subscription_count() and self.count_publishers("/carla/ego/command"))
        if self.strict_peers:  # mpc_node and policy_node matched, so tick 0 is neither dropped nor late
            ok = ok and all(
                pub.get_subscription_count()
                for pub in (self.route_pub, self.odom_pub, self.objects_pub, self.context_pub)
            )
            ok = (
                ok
                and self.count_publishers("/autonomy/policy_action")
                and self.count_subscribers("/autonomy/features")
            )
        return ok

    def _wait_for_peers(self) -> None:
        end = time.monotonic() + PEER_WAIT_S
        while time.monotonic() < end:
            if self._peers_ready():
                time.sleep(PEER_SETTLE_S)
                return
            time.sleep(0.05)
        self.get_logger().warning("no /clock subscriber or command publisher found; ticking anyway")

    def _scenario(self):
        routes = self.routes or str(Path(self.sensors).parents[1] / "scenarios/carla_routes.yaml")
        sc = make_scenario(self.scenario_name, self.seed)
        return carla_scenario(
            sc, self.scenario_name, self.seed, self.town or None, Path(routes), fallback=True
        )

    def run(self) -> None:
        sim = None
        try:
            sc = self._scenario()
            sim = CarlaSim(
                CarlaSimConfig(
                    town=sc.carla_route["town"],
                    quality=self.quality,
                    sensors=Path(self.sensors) if self.sensors else None,
                    seed=self.seed,
                    no_rendering=self.no_rendering,
                )
            )
            self.driver = B.EpisodeDriver(sim, sc)  # resets the sim (settle ticks, spawn) in this thread
            route = RouteInfo()
            C.fill_route(route, self.driver.route)
            self.route_pub.publish(route)
            self._wait_for_peers()
            loop = TickLoop(
                self.driver.step, self.publish, self.mailbox, self.command_timeout, self.warmup_ticks
            )
            self.stats = loop.run(self.max_ticks or None)
            self._write_result()
        except Exception:  # noqa: BLE001 - architecture.md section 8: a CARLA RPC failure ends the run non-zero
            self.get_logger().error(traceback.format_exc())
            self.failed = True
        finally:
            if sim is not None:
                sim.close()  # restores asynchronous mode and destroys our actors
            self.done.set()

    def _write_result(self) -> None:
        summary = self.driver.summary(self.scenario_name, self.seed) | {
            "tick_stats": asdict(self.stats),
            "skipped_sensor_frames": self.skipped,
        }
        self.get_logger().info(f"episode end: {summary}")
        if self.result_path:
            Path(self.result_path).write_text(json.dumps(summary, indent=2))


def main() -> None:
    rclpy.init()
    node = CarlaBridge()
    threading.Thread(target=node.run, daemon=True).start()
    try:
        while rclpy.ok() and not node.done.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(1 if node.failed else 0)
