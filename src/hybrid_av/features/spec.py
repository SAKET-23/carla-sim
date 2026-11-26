"""The 24-D state vector; the table is Plan/02-system-design/state-vector.md."""

from dataclasses import dataclass

import numpy as np

HALF_PI = float(np.pi / 2)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    index: int
    name: str
    unit: str
    lo: float  # physical value mapped to -1
    hi: float  # physical value mapped to +1
    source_gt: str
    source_perception: str


_ROWS = (
    ("vx", "m/s", 0, 30, "ego.vx", "/localization/ego_state twist"),
    ("vy", "m/s", -3, 3, "ego.vy", "same"),
    ("yaw_rate", "rad/s", -1.5, 1.5, "ego.r", "IMU gyro z"),
    ("lateral_offset", "m", -3.5, 3.5, "Route.project(ego).lateral", "same, using localised pose"),
    ("heading_error", "rad", -HALF_PI, HALF_PI, "wrap(ego.psi - route.psi(s))", "same"),
    ("target_curvature", "1/m", -0.1, 0.1, "max-abs signed curvature over the next max(20, 2*vx) m", "same"),
    ("friction_est", "-", 0, 1.2, "FrictionEstimator.update", "same (estimator uses ego state only)"),
    (
        "surface_roughness",
        "0..1",
        0,
        1,
        "context.surface_roughness: scenario-set surface-condition index (0 smooth and dry, 1 rough or standing water)",
        "same value from the scenario definition, in both modes",
    ),
    ("grade", "rad", -0.15, 0.15, "context.road_grade", "same, from map"),
    (
        "lead_dist",
        "m",
        0,
        80,
        "bumper-to-bumper gap to nearest object in corridor, same direction (heading diff < 45 deg)",
        "from tracks",
    ),
    ("lead_rel_speed", "m/s", -15, 15, "lead_speed - ego.vx (along route)", "from tracks"),
    ("lead_ttc", "s", 0, 10, "gap / max(-rel_speed, eps)", "from tracks"),
    (
        "lead_accel",
        "m/s^2",
        -6,
        3,
        "lead longitudinal acceleration",
        "finite difference of track speed, low-pass tau = 0.3 s",
    ),
    ("lead_lateral_offset", "m", -3.5, 3.5, "lead's lateral offset from route centreline", "from tracks"),
    (
        "ped_dist",
        "m",
        0,
        50,
        "Euclidean distance to nearest pedestrian within 50 m and +-60 deg of the route heading",
        "from tracks",
    ),
    (
        "ped_bearing",
        "rad",
        -HALF_PI,
        HALF_PI,
        "bearing of that pedestrian relative to ego heading",
        "from tracks",
    ),
    (
        "cross_veh_dist",
        "m",
        0,
        80,
        "distance along the crossing vehicle's own path to the conflict point, for the vehicle reaching it soonest",
        "from tracks (constant-velocity path)",
    ),
    ("cross_veh_speed", "m/s", 0, 20, "that vehicle's speed", "from tracks"),
    ("cross_veh_ttc", "s", 0, 10, "cross_veh_dist / max(cross_veh_speed, eps)", "from tracks"),
    (
        "traffic_light_state",
        "-",
        0,
        3,
        "context.light_state as 0 none, 1 green, 2 yellow, 3 red",
        "from map + light state",
    ),
    (
        "intersection_gap",
        "s",
        0,
        10,
        "cross_veh_ttc - ego_time_to_clear at a junction; no junction within 50 m -> hi",
        "from tracks",
    ),
    ("speed_limit", "m/s", 0, 30, "context.speed_limit", "from map"),
    ("lane_width", "m", 2.5, 5, "context.lane_width", "from map"),
    (
        "stop_dist",
        "m",
        0,
        100,
        "min(distance to stop line if light is yellow or red, distance to stop sign if not yet served)",
        "from map",
    ),
)

FEATURES: tuple[FeatureSpec, ...] = tuple(
    FeatureSpec(i, name, unit, float(lo), float(hi), gt, perc)
    for i, (name, unit, lo, hi, gt, perc) in enumerate(_ROWS)
)
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)

_LO = np.array([f.lo for f in FEATURES])
_HI = np.array([f.hi for f in FEATURES])


def normalize(raw: np.ndarray) -> np.ndarray:
    """(24,) or (B, 24) physical -> [-1, 1], clipped."""
    return np.clip(2.0 * (raw - _LO) / (_HI - _LO) - 1.0, -1.0, 1.0)


def denormalize(z: np.ndarray) -> np.ndarray:
    return (np.asarray(z) + 1.0) / 2.0 * (_HI - _LO) + _LO


I_VX = 0
I_VY = 1
I_YAW_RATE = 2
I_LATERAL_OFFSET = 3
I_HEADING_ERROR = 4
I_TARGET_CURVATURE = 5
I_FRICTION_EST = 6
I_SURFACE_ROUGHNESS = 7
I_GRADE = 8
I_LEAD_DIST = 9
I_LEAD_REL_SPEED = 10
I_LEAD_TTC = 11
I_LEAD_ACCEL = 12
I_LEAD_LATERAL_OFFSET = 13
I_PED_DIST = 14
I_PED_BEARING = 15
I_CROSS_VEH_DIST = 16
I_CROSS_VEH_SPEED = 17
I_CROSS_VEH_TTC = 18
I_TRAFFIC_LIGHT_STATE = 19
I_INTERSECTION_GAP = 20
I_SPEED_LIMIT = 21
I_LANE_WIDTH = 22
I_STOP_DIST = 23
