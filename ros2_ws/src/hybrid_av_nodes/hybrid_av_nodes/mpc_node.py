"""mpc_node: owns the single AutonomyStack; publishes features, waits for the policy action, publishes the command."""

import math
import time

import rclpy
from hybrid_av_msgs.msg import (
    FeatureVector,
    MpcDiagnostics,
    PolicyAction,
    ReferenceTrajectory,
    RouteInfo,
    TrackArray,
    TrafficContext,
    VehicleCommand,
)
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from hybrid_av.config import load_dataclass
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import Observation
from hybrid_av.types import PolicyAction as LibAction
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from hybrid_av_nodes import convert as C
from hybrid_av_nodes.mpc_core import MpcCore
from hybrid_av_nodes.stamp_sync import StampSync
from hybrid_av_nodes.tick_loop import StampMailbox

_INPUTS = {  # mode -> (ego, objects) topics; the traffic context always comes from the bridge
    "ground_truth": ("/carla/ego/ground_truth/odom", "/carla/ground_truth/objects"),
    "perception": ("/localization/ego_state", "/perception/tracks3d"),
}
_NEUTRAL = LibAction(1.0, 0.0, 0.0)  # held until the first policy action arrives


def _load(cls, path):
    return load_dataclass(cls, path) if path else cls()  # empty path: dataclass defaults


class MpcNode(Node):
    def __init__(self) -> None:
        super().__init__("mpc_node")
        p = self.declare_parameter
        self.mode = p("mode", "ground_truth").value
        if self.mode not in _INPUTS:
            raise ValueError(f"mode must be ground_truth or perception, got {self.mode!r}")
        vehicle, mpc, ref = p("vehicle", "").value, p("mpc", "").value, p("reference", "").value
        mpc_cfg = _load(MPCConfig, mpc)
        stack = AutonomyStack(
            VehicleParams.from_yaml(vehicle) if vehicle else VehicleParams(),
            LongitudinalMap(), mpc_cfg, _load(ReferenceConfig, ref),
        )  # fmt: skip
        self.dt = mpc_cfg.dt
        self.core = MpcCore(stack, int(p("policy_period", 2).value))
        self.action_timeout = float(p("action_timeout", 1.0).value)
        self.peer_wait_s = float(
            p("peer_wait_s", 0.0).value
        )  # launch sets it; 0 = tick without waiting for peers
        self.unwrap = C.HeadingUnwrapper()
        self.sync = StampSync(("ego", "objects", "ctx"))
        self.actions: StampMailbox[LibAction] = StampMailbox()
        self.last_action = _NEUTRAL
        self.late_actions = 0
        self.t_first: float | None = None
        self.have_route = False

        reliable = lambda depth: QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=depth)
        ego_topic, objects_topic = _INPUTS[self.mode]
        for key, msg_type, topic in (
            ("ego", Odometry, ego_topic),
            ("objects", TrackArray, objects_topic),
            ("ctx", TrafficContext, "/carla/ego/traffic_context"),
        ):
            self.create_subscription(msg_type, topic, lambda m, k=key: self._on_input(k, m), reliable(10))
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1
        )
        self.create_subscription(RouteInfo, "/carla/ego/route", self._on_route, latched)
        # the action arrives while an input callback is blocked waiting for it: separate callback group
        self.create_subscription(
            PolicyAction, "/autonomy/policy_action", self._on_action, reliable(1),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )  # fmt: skip
        self.features_pub = self.create_publisher(FeatureVector, "/autonomy/features", reliable(1))
        self.diag_pub = self.create_publisher(MpcDiagnostics, "/autonomy/mpc_diagnostics", reliable(10))
        self.ref_pub = self.create_publisher(ReferenceTrajectory, "/autonomy/reference", reliable(1))
        self.cmd_pub = self.create_publisher(VehicleCommand, "/carla/ego/command", reliable(1))

    def _on_route(self, m: RouteInfo) -> None:
        self.core.on_route(C.route_from_msg(m))
        self.t_first, self.have_route = None, True

    def _on_action(self, m: PolicyAction) -> None:
        self.actions.put(C.stamp_to_sec(m.header.stamp), C.action_from_msg(m))

    def _on_input(self, key: str, m) -> None:
        if not self.have_route:
            self.get_logger().warning("observation before the route: dropped", throttle_duration_sec=2.0)
            return
        t = C.stamp_to_sec(m.header.stamp)
        s = self.sync.add(key, t, m)
        if s is not None:
            self._on_set(t, s)

    def _wait_for_peers(self) -> None:
        # matched endpoints, not the graph: tick 0 must reach policy_node and the command must reach the tick owner
        end = time.monotonic() + self.peer_wait_s
        ready = lambda: (
            self.features_pub.get_subscription_count() > 0
            and self.cmd_pub.get_subscription_count() > 0
            and self.count_publishers("/autonomy/policy_action") > 0
        )
        while not ready() and time.monotonic() < end:
            time.sleep(0.02)
        if self.peer_wait_s and not ready():
            self.get_logger().warning("policy/command peers not matched in time; ticking anyway")

    def _on_set(self, t: float, s: dict) -> None:
        obs = Observation(
            C.ego_from_odometry(s["ego"], self.unwrap),
            C.objects_from_track_array(s["objects"]),
            C.context_from_msg(s["ctx"]),
            math.nan,  # friction_true is never available from ROS
            self.mode,
        )
        if self.t_first is None:
            self.t_first = t
            self._wait_for_peers()
        tick = round((t - self.t_first) / self.dt)
        frame = self.core.features(obs)
        stamp = s["ego"].header.stamp

        def stamped(msg):
            msg.header.stamp = stamp
            return msg

        fm = stamped(FeatureVector())
        C.fill_features(fm, frame, self.mode)
        self.features_pub.publish(fm)
        if self.core.needs_action(tick):
            a = self.actions.wait_for(t, self.action_timeout)
            if a is None:
                self.late_actions += 1
                self.get_logger().warning(f"late policy action at t={t:.2f}; holding the previous one")
            else:
                self.last_action = a
        out = self.core.act(obs, self.last_action, frame)
        dm = stamped(MpcDiagnostics())
        C.fill_diagnostics(dm, out.mpc)
        self.diag_pub.publish(dm)
        rm = stamped(ReferenceTrajectory())
        C.fill_reference(rm, out.reference, self.dt)
        self.ref_pub.publish(rm)
        cm = stamped(VehicleCommand())  # last: the tick owner proceeds as soon as it arrives
        C.fill_command(cm, out.command)
        self.cmd_pub.publish(cm)


def main() -> None:
    rclpy.init()
    node = MpcNode()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
