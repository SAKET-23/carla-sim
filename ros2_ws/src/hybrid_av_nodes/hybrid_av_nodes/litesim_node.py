"""LiteSim as a ROS 2 tick owner: stands in for carla_bridge on the Mac (architecture.md section 5)."""

import json
import threading
import time
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
from tf2_ros import TransformBroadcaster

from hybrid_av.eval.driving_score import driving_score
from hybrid_av.eval.runner import INFRACTION_KEYS
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from hybrid_av_nodes import convert as C
from hybrid_av_nodes.tick_loop import CommandMailbox, TickLoop

PEER_SETTLE_S = 0.5
PEER_WAIT_S = 30.0  # wall-clock wait for a /clock subscriber and a command publisher before ticking


class LiteSimNode(Node):
    def __init__(self) -> None:
        super().__init__("litesim_node")
        p = self.declare_parameter
        self.scenario_name = p("scenario", "nominal").value
        self.seed = int(p("seed", 0).value)
        self.dt = float(p("dt", 0.05).value)
        self.command_timeout = float(p("command_timeout", 1.0).value)
        self.max_ticks = int(p("max_ticks", 0).value)
        self.result_path = p("result_path", "").value
        self.strict_peers = bool(p("strict_peers", False).value)  # launch sets it: wait for the whole stack
        self.rng = np.random.default_rng(int(p("wheel_odom_seed", 0).value))

        reliable = lambda depth: QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=depth)
        self.clock_pub = self.create_publisher(Clock, "/clock", reliable(10))
        self.odom_pub = self.create_publisher(Odometry, "/carla/ego/ground_truth/odom", reliable(10))
        self.objects_pub = self.create_publisher(TrackArray, "/carla/ground_truth/objects", reliable(10))
        self.context_pub = self.create_publisher(TrafficContext, "/carla/ego/traffic_context", reliable(10))
        self.wheel_pub = self.create_publisher(Odometry, "/carla/ego/wheel_odom", qos_profile_sensor_data)
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1
        )
        self.route_pub = self.create_publisher(RouteInfo, "/carla/ego/route", latched)
        self.tf = TransformBroadcaster(self)
        self.mailbox = CommandMailbox()
        self.create_subscription(VehicleCommand, "/carla/ego/command", self._on_command, reliable(1))

        self.sim = LiteSim(VehicleParams(), LongitudinalMap(), self.dt)
        self.scenario = make_scenario(self.scenario_name, self.seed)
        self.obs = self.sim.reset(self.scenario)
        self.route = self.sim.route()
        self.infractions = dict.fromkeys(INFRACTION_KEYS, 0)
        self.info = None
        self._first = True
        self.stats = None
        self.done = threading.Event()

    def _on_command(self, m: VehicleCommand) -> None:
        self.mailbox.put(C.stamp_to_sec(m.header.stamp), C.command_from_msg(m))

    # --- TickLoop callbacks ----------------------------------------------------
    def step(self, cmd):
        if self._first:  # tick 0 publishes the reset observation, so command k answers observation k
            self._first = False
            return self.obs.ego.t, False
        self.obs, info = self.sim.step(cmd)
        self.info = info
        events = {"red_light": info.red_light_violation, "stop_sign": info.stop_sign_violation}
        if info.collision != "none":
            events[f"collision_{info.collision}"] = True
        for key, hit in events.items():
            self.infractions[key] += bool(hit)
        if info.off_road:
            self.infractions["off_road_m"] += self.obs.ego.vx * self.dt
        end = info.route_completed or info.route_deviation or info.blocked or info.timeout
        return self.obs.ego.t, end or info.collision != "none"

    def publish(self, t: float) -> None:
        ego, stamp = self.obs.ego, Clock()
        C.fill_stamp(stamp.clock, t)
        self.clock_pub.publish(stamp)  # first within the tick (ECC-20)

        def stamped(msg):
            msg.header.stamp = stamp.clock
            return msg

        odom = stamped(Odometry())
        C.fill_odometry(odom, ego, "map", "base_link")
        self.odom_pub.publish(odom)
        objs = stamped(TrackArray())
        objs.header.frame_id = "map"
        C.fill_track_array(objs, self.obs.objects, Track)
        self.objects_pub.publish(objs)
        ctx = stamped(TrafficContext())
        ctx.header.frame_id = "map"
        C.fill_context(ctx, self.obs.context)
        self.context_pub.publish(ctx)
        wheel = stamped(Odometry())  # twist only: GT body velocity, N(0, 0.05) noise and +1 % scale
        wheel.header.frame_id = wheel.child_frame_id = "base_link"
        n = self.rng.normal(0.0, 0.05, 2)
        wheel.twist.twist.linear.x, wheel.twist.twist.linear.y = 1.01 * ego.vx + n[0], 1.01 * ego.vy + n[1]
        wheel.twist.twist.angular.z = ego.r
        self.wheel_pub.publish(wheel)
        tf = stamped(TransformStamped())
        tf.header.frame_id, tf.child_frame_id = "map", "ground_truth_base_link"
        tf.transform.translation.x, tf.transform.translation.y = ego.x, ego.y
        C.fill_quat_from_yaw(tf.transform.rotation, ego.psi)
        self.tf.sendTransform(tf)

    # --- run ---------------------------------------------------------------------
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
                time.sleep(PEER_SETTLE_S)  # the reader side learns of our writer slightly later than we of it
                return
            time.sleep(0.05)
        self.get_logger().warning("no /clock subscriber or command publisher found; ticking anyway")

    def run(self) -> None:
        try:
            route = RouteInfo()
            C.fill_route(route, self.route)
            self.route_pub.publish(route)
            self._wait_for_peers()
            loop = TickLoop(self.step, self.publish, self.mailbox, self.command_timeout)
            self.stats = loop.run(self.max_ticks or None)
            self._write_result()
        finally:
            self.done.set()

    def _write_result(self) -> None:
        info = self.info
        done = info is not None and info.route_completed
        completion = 1.0 if done else min(1.0, (info.route_s if info else 0.0) / self.route.length)
        inf = dict(self.infractions)
        inf["timeout"] = int(bool(info and info.timeout and not info.route_completed))
        inf["blocked"] = int(bool(info and info.blocked))
        summary = {
            "scenario": self.scenario_name,
            "seed": self.seed,
            "route_completion": completion,
            "infractions": inf,
            "driving_score": driving_score(completion, inf, self.route.length),
            "duration": self.obs.ego.t,
            "tick_stats": asdict(self.stats),
        }
        self.get_logger().info(f"episode end: {summary}")
        if self.result_path:
            Path(self.result_path).write_text(json.dumps(summary, indent=2))


def main() -> None:
    rclpy.init()
    node = LiteSimNode()
    threading.Thread(target=node.run, daemon=True).start()
    try:
        while rclpy.ok() and not node.done.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
