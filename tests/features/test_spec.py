import numpy as np

from hybrid_av.features.spec import (
    FEATURE_NAMES,
    FEATURES,
    I_CROSS_VEH_TTC,
    I_FRICTION_EST,
    denormalize,
    normalize,
)

NAMES = [
    "vx",
    "vy",
    "yaw_rate",
    "lateral_offset",
    "heading_error",
    "target_curvature",
    "friction_est",
    "surface_roughness",
    "grade",
    "lead_dist",
    "lead_rel_speed",
    "lead_ttc",
    "lead_accel",
    "lead_lateral_offset",
    "ped_dist",
    "ped_bearing",
    "cross_veh_dist",
    "cross_veh_speed",
    "cross_veh_ttc",
    "traffic_light_state",
    "intersection_gap",
    "speed_limit",
    "lane_width",
    "stop_dist",
]


def test_table():
    assert list(FEATURE_NAMES) == NAMES and [f.index for f in FEATURES] == list(range(24))
    assert I_FRICTION_EST == 6 and I_CROSS_VEH_TTC == 18


def test_normalize_endpoints():
    lo = np.array([f.lo for f in FEATURES])
    hi = np.array([f.hi for f in FEATURES])
    np.testing.assert_allclose(normalize(lo), -1)
    np.testing.assert_allclose(normalize(hi), 1)
    np.testing.assert_allclose(normalize((lo + hi) / 2), 0, atol=1e-12)
    np.testing.assert_allclose(normalize(hi + 100), 1)
    assert normalize(np.tile(lo, (5, 1))).shape == (5, 24)
    np.testing.assert_allclose(denormalize(normalize((lo + hi) / 3)), (lo + hi) / 3)
