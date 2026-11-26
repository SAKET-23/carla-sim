"""Message <-> library dataclass conversions (interfaces.md section 14).

Pure core: no rclpy and no message imports. Functions read anything with the same attributes as the ROS
message; "fill" functions write into a message passed in. float32 message fields are rounded to float32
on write, which is what a real message stores, so the Mac tests (SimpleNamespace trees) see the same values.
"""

import math
from collections.abc import Callable, Sequence
from types import SimpleNamespace as NS
from typing import Any

import numpy as np

from hybrid_av.features.extractor import FeatureFrame
from hybrid_av.features.spec import I_HEADING_ERROR, I_LATERAL_OFFSET, I_STOP_DIST
from hybrid_av.geometry import unwrap_near
from hybrid_av.mpc.controller import MPCResult, Reference
from hybrid_av.perception.base import Box3D, RadarDetection
from hybrid_av.planning.route import Route, RouteBuilder
from hybrid_av.types import (
    ActuatorCommand,
    EgoState,
    ObjectClass,
    ObjectState,
    Observation,
    PolicyAction,
    TrafficContext,
    TrafficLightState,
)

# float32 fields per message type (interfaces.md section 14)
F32_FIELDS: dict[str, tuple[str, ...]] = {
    "FeatureVector": ("raw", "normalized", "mu_est", "route_s"),
    "PolicyAction": ("alpha_v", "delta_y", "brake"),
    "ReferenceTrajectory": ("dt", "x", "y", "psi", "v"),
    "MpcDiagnostics": ("solve_time_ms", "cost", "slack_friction_max", "slack_obstacle_max", "delta", "ax"),
    "VehicleCommand": ("steer", "throttle", "brake"),
    "TrafficContext": (
        "dist_to_stop_line",
        "dist_to_stop_sign",
        "dist_to_junction",
        "speed_limit",
        "lane_width",
        "road_grade",
        "surface_roughness",
    ),
    "Track": ("yaw", "pos_cov"),
    "RouteInfo": ("x", "y", "speed_limit", "lane_width"),
}

_POINT_FLOAT32 = 7  # sensor_msgs/PointField.FLOAT32
_POINT_DTYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def _f32(v: float) -> float:
    return float(np.float32(v))


def _f32s(a) -> list[float]:
    # a plain list: rclpy's float32[] setters reject float32 ndarrays
    return np.asarray(a, dtype=np.float32).tolist()


# --- time and orientation --------------------------------------------------
def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec / 1e9


def fill_stamp(stamp, t: float) -> None:
    sec = math.floor(t)
    nanosec = round((t - sec) * 1e9)
    if nanosec >= 1_000_000_000:
        sec, nanosec = sec + 1, nanosec - 1_000_000_000
    stamp.sec, stamp.nanosec = sec, nanosec


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def fill_quat_from_yaw(q, yaw: float) -> None:
    q.x = q.y = 0.0
    q.z, q.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class HeadingUnwrapper:
    """Continuous heading across +-pi. `prev` seeds the first value (default: the first yaw is returned as is)."""

    def __init__(self, prev: float | None = None) -> None:
        self.prev = prev

    def __call__(self, yaw: float) -> float:
        self.prev = yaw if self.prev is None else unwrap_near(yaw, self.prev)
        return self.prev


# --- ego ---------------------------------------------------------------------
def ego_from_odometry(msg, unwrap: HeadingUnwrapper) -> EgoState:
    p, tw = msg.pose.pose, msg.twist.twist
    return EgoState(
        t=stamp_to_sec(msg.header.stamp),
        x=p.position.x,
        y=p.position.y,
        psi=unwrap(yaw_from_quat(p.orientation)),
        vx=tw.linear.x,
        vy=tw.linear.y,
        r=tw.angular.z,
    )


def fill_odometry(msg, ego: EgoState, frame_id: str, child_frame_id: str) -> None:
    msg.header.frame_id, msg.child_frame_id = frame_id, child_frame_id
    p, tw = msg.pose.pose, msg.twist.twist
    p.position.x, p.position.y, p.position.z = ego.x, ego.y, 0.0
    fill_quat_from_yaw(p.orientation, ego.psi)
    tw.linear.x, tw.linear.y, tw.angular.z = ego.vx, ego.vy, ego.r


# --- objects -----------------------------------------------------------------
def objects_from_track_array(msg) -> tuple[ObjectState, ...]:
    out = []
    for t in msg.tracks:
        cov = np.asarray(t.pos_cov, dtype=np.float64).reshape(2, 2)
        out.append(
            ObjectState(
                id=int(t.id), cls=ObjectClass(t.cls),
                x=t.position.x, y=t.position.y, z=t.position.z, yaw=t.yaw,
                length=t.size.x, width=t.size.y, height=t.size.z,
                vx=t.velocity.x, vy=t.velocity.y,
                pos_cov=cov if cov.any() else None,  # all zeros = ground truth
            )
        )  # fmt: skip
    return tuple(out)


def fill_track_array(msg, objects: Sequence[ObjectState], track_factory: Callable[[], Any]) -> None:
    tracks = []
    for o in objects:
        t = track_factory()
        t.id, t.cls = o.id, int(o.cls)
        t.position.x, t.position.y, t.position.z = o.x, o.y, o.z
        t.yaw = _f32(o.yaw)
        t.size.x, t.size.y, t.size.z = o.length, o.width, o.height
        t.velocity.x, t.velocity.y, t.velocity.z = o.vx, o.vy, 0.0
        t.pos_cov = _f32s(np.zeros(4) if o.pos_cov is None else np.asarray(o.pos_cov).reshape(4))
        tracks.append(t)
    msg.tracks = tracks


def context_from_msg(msg) -> TrafficContext:
    return TrafficContext(
        TrafficLightState(msg.light_state), msg.dist_to_stop_line, msg.dist_to_stop_sign,
        msg.dist_to_junction, msg.speed_limit, msg.lane_width, msg.road_grade, msg.surface_roughness,
    )  # fmt: skip


def fill_context(msg, ctx: TrafficContext) -> None:
    msg.light_state = int(ctx.light_state)
    for f in F32_FIELDS["TrafficContext"]:
        setattr(msg, f, _f32(getattr(ctx, f)))


# --- route -------------------------------------------------------------------
def route_from_msg(msg) -> Route:
    xy = np.column_stack([np.asarray(msg.x, np.float64), np.asarray(msg.y, np.float64)])
    return RouteBuilder.from_waypoints(
        xy, np.asarray(msg.speed_limit, np.float64), np.asarray(msg.lane_width, np.float64), spacing=0.5
    )


def fill_route(msg, route: Route) -> None:
    msg.x, msg.y = _f32s(route.x), _f32s(route.y)
    msg.speed_limit, msg.lane_width = _f32s(route.speed_limit), _f32s(route.lane_width)


# --- policy action, command, features, diagnostics, reference ---------------------
def action_from_msg(msg) -> PolicyAction:
    return PolicyAction(msg.alpha_v, msg.delta_y, msg.brake)


def fill_action(msg, a: PolicyAction, policy_hash: str) -> None:
    msg.alpha_v, msg.delta_y, msg.brake = _f32(a.alpha_v), _f32(a.delta_y), _f32(a.brake)
    msg.policy_hash = policy_hash


def command_from_msg(msg) -> ActuatorCommand:
    return ActuatorCommand(msg.steer, msg.throttle, msg.brake)


def fill_command(msg, cmd: ActuatorCommand) -> None:
    msg.steer, msg.throttle, msg.brake = _f32(cmd.steer), _f32(cmd.throttle), _f32(cmd.brake)


def fill_features(msg, frame, source: str) -> None:
    msg.raw, msg.normalized = _f32s(frame.raw), _f32s(frame.z)
    msg.mu_est, msg.route_s, msg.source = _f32(frame.mu_est), _f32(frame.route_s), source


def frame_from_feature_msg(msg) -> FeatureFrame:
    """dist_to_junction is not on the wire: slot 23 (min stop distance, capped at 100 m) stands in for it."""
    raw = np.asarray(msg.raw, dtype=np.float64)
    stop = raw[I_STOP_DIST]
    return FeatureFrame(
        raw, np.asarray(msg.normalized, dtype=np.float32), msg.route_s, raw[I_LATERAL_OFFSET],
        raw[I_HEADING_ERROR], msg.mu_est, None, None, stop if stop < 100.0 else math.inf, math.inf,
    )  # fmt: skip


def fill_diagnostics(msg, res: MPCResult) -> None:
    msg.status = res.status
    msg.solve_time_ms = _f32(res.solve_time * 1e3)
    msg.cost = _f32(res.cost)
    msg.slack_friction_max, msg.slack_obstacle_max = (
        _f32(res.slack_friction_max),
        _f32(res.slack_obstacle_max),
    )
    msg.delta, msg.ax = _f32(res.control.delta), _f32(res.control.ax)


def fill_reference(msg, ref: Reference, dt: float) -> None:
    msg.dt = _f32(dt)
    msg.x, msg.y, msg.psi, msg.v = _f32s(ref.x), _f32s(ref.y), _f32s(ref.psi), _f32s(ref.v)


# --- PointCloud2 ---------------------------------------------------------------
def _cloud(msg) -> np.ndarray:
    dt = np.dtype(
        {
            "names": [f.name for f in msg.fields],
            "formats": [_POINT_DTYPES[f.datatype] for f in msg.fields],
            "offsets": [f.offset for f in msg.fields],
            "itemsize": msg.point_step,
        }
    )
    return np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)


def xyzi_from_pointcloud2(msg) -> np.ndarray:
    c = _cloud(msg)
    return np.column_stack([c["x"], c["y"], c["z"], c["intensity"]]).astype(np.float32, copy=False)


def radar_from_pointcloud2(msg, sensor_id: str) -> list[RadarDetection]:
    c = _cloud(msg)
    cols = [c[k].astype(np.float64).tolist() for k in ("depth", "azimuth", "altitude", "velocity")]
    return [RadarDetection(d, az, al, v, sensor_id) for d, az, al, v in zip(*cols)]


def fill_pointcloud2(
    msg, arr: np.ndarray, field_names: Sequence[str], field_factory: Callable[[], Any]
) -> None:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    fields = []
    for i, name in enumerate(field_names):
        f = field_factory()
        f.name, f.offset, f.datatype, f.count = name, 4 * i, _POINT_FLOAT32, 1
        fields.append(f)
    msg.fields = fields
    msg.height, msg.width, msg.is_bigendian, msg.is_dense = 1, len(arr), False, True
    msg.point_step = 4 * len(field_names)
    msg.row_step = msg.point_step * len(arr)
    msg.data = arr.tobytes()


# --- 3D detections -----------------------------------------------------------------
def boxes_from_detection_array(msg) -> list[Box3D]:
    out = []
    for d in msg.detections:
        c, s, h = d.bbox.center, d.bbox.size, d.results[0].hypothesis
        out.append(
            Box3D(c.position.x, c.position.y, c.position.z, s.x, s.y, s.z, yaw_from_quat(c.orientation),
                  ObjectClass[h.class_id], h.score)
        )  # fmt: skip
    return out


def fill_detection_array(msg, boxes: Sequence[Box3D], det_factory: Callable[[], Any]) -> None:
    """`det_factory()` returns a detection whose `results` already holds one hypothesis."""
    dets = []
    for b in boxes:
        d = det_factory()
        c = d.bbox.center
        c.position.x, c.position.y, c.position.z = b.x, b.y, b.z
        fill_quat_from_yaw(c.orientation, b.yaw)
        d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = b.length, b.width, b.height
        d.results[0].hypothesis.class_id = b.cls.name
        d.results[0].hypothesis.score = _f32(b.score)
        dets.append(d)
    msg.detections = dets


# --- msgns-style builders: SimpleNamespace trees shaped like the ROS messages -----------
def ns_header():
    return NS(stamp=NS(sec=0, nanosec=0), frame_id="")


def ns_odometry():
    xyz = lambda: NS(x=0.0, y=0.0, z=0.0)
    return NS(
        header=ns_header(), child_frame_id="",
        pose=NS(pose=NS(position=xyz(), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))),
        twist=NS(twist=NS(linear=xyz(), angular=xyz())),
    )  # fmt: skip


def ns_track():
    xyz = lambda: NS(x=0.0, y=0.0, z=0.0)
    return NS(id=0, cls=0, position=xyz(), yaw=0.0, size=xyz(), velocity=xyz(), pos_cov=[0.0] * 4)


def ns_track_array():
    return NS(header=ns_header(), tracks=[])


def ns_traffic_context():
    return NS(header=ns_header(), light_state=0, **dict.fromkeys(F32_FIELDS["TrafficContext"], 0.0))


def ns_route_info():
    return NS(header=ns_header(), x=[], y=[], speed_limit=[], lane_width=[])


def ns_feature_vector():
    return NS(header=ns_header(), raw=[0.0] * 24, normalized=[0.0] * 24, mu_est=0.0, route_s=0.0, source="")


def ns_policy_action():
    return NS(header=ns_header(), alpha_v=0.0, delta_y=0.0, brake=0.0, policy_hash="")


def ns_pointcloud2():
    return NS(header=ns_header(), height=0, width=0, fields=[], is_bigendian=False, point_step=0,
              row_step=0, data=b"", is_dense=True)  # fmt: skip


def ns_field():
    return NS(name="", offset=0, datatype=0, count=1)


# --- parity helpers (ECC-22): what the ROS path does to an observation ----------------
def roundtrip_observation(
    obs: Observation, source: str = "ground_truth", unwrap: HeadingUnwrapper | None = None
) -> Observation:
    """Apply what mpc_node sees: Odometry quaternion and stamp, float32 Track and TrafficContext fields."""
    om = ns_odometry()
    fill_odometry(om, obs.ego, "map", "base_link")
    fill_stamp(om.header.stamp, obs.ego.t)
    ego = ego_from_odometry(om, unwrap or HeadingUnwrapper(prev=obs.ego.psi))
    tm = ns_track_array()
    fill_track_array(tm, obs.objects, ns_track)
    cm = ns_traffic_context()
    fill_context(cm, obs.context)
    return Observation(ego, objects_from_track_array(tm), context_from_msg(cm), math.nan, source)


def roundtrip_action(a: PolicyAction) -> PolicyAction:
    m = ns_policy_action()
    fill_action(m, a, "")
    return action_from_msg(m)


def roundtrip_route(route: Route) -> Route:
    m = ns_route_info()
    fill_route(m, route)
    return route_from_msg(m)
