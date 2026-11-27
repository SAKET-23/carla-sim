import numpy as np
import pytest

from hybrid_av.features.extractor import FeatureExtractor
from hybrid_av.features.spec import *  # I_* constants, FEATURE_NAMES
from hybrid_av.planning.route import RouteBuilder
from hybrid_av.types import (
    Control,
    EgoState,
    ObjectClass,
    ObjectState,
    Observation,
    TrafficContext,
)
from hybrid_av.types import TrafficLightState as L
from hybrid_av.vehicle.params import VehicleParams


def _obs(
    ego, objects=(), light=L.NONE, dist_to_stop_line=np.inf, dist_to_junction=np.inf, roughness=0.1, grade=0.0
):
    ctx = TrafficContext(light, dist_to_stop_line, np.inf, dist_to_junction, 13.89, 3.5, grade, roughness)
    return Observation(ego, tuple(objects), ctx, 1.0, "ground_truth")


def test_ego_path_group():
    fx = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89))
    f = fx.extract(_obs(ego=EgoState(0, 50, 1.0, 0.1, 10, 0.2, 0.05)), Control(0, 0), 0.05)
    np.testing.assert_allclose(
        f.raw[[I_VX, I_VY, I_YAW_RATE, I_LATERAL_OFFSET, I_HEADING_ERROR, I_TARGET_CURVATURE]],
        [10, 0.2, 0.05, 1.0, 0.1, 0.0],
        atol=1e-6,
    )
    assert f.z.dtype == np.float32 and np.all(np.abs(f.z) <= 1)


def test_curvature_lookahead():
    r = RouteBuilder.concat(RouteBuilder.straight(60, 13.89), RouteBuilder.arc(30, np.pi / 2, 13.89))
    f = FeatureExtractor(VehicleParams(), r).extract(
        _obs(ego=EgoState(0, 50, 0, 0, 10, 0, 0)), Control(0, 0), 0.05
    )
    assert abs(f.raw[I_TARGET_CURVATURE] - 1 / 30) < 1e-3  # arc starts 10 m ahead, look-ahead 20 m


def test_road_group():
    f = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89)).extract(
        _obs(ego=EgoState(0, 50, 0, 0, 10, 0, 0), roughness=0.8, grade=0.05), Control(0, 0), 0.05
    )
    assert f.raw[I_FRICTION_EST] == 1.0 and f.raw[I_SURFACE_ROUGHNESS] == 0.8 and f.raw[I_GRADE] == 0.05


def test_lead_group():
    lead = ObjectState(7, ObjectClass.CAR, 70, 0.5, 0, 0, 4.5, 1.9, 1.5, 8.0, 0.0)
    f = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89)).extract(
        _obs(ego=EgoState(0, 50, 0, 0, 10, 0, 0), objects=(lead,)), Control(0, 0), 0.05
    )
    gap = 70 - 2.25 - 51.7  # 16.05 m
    np.testing.assert_allclose(
        f.raw[[I_LEAD_DIST, I_LEAD_REL_SPEED, I_LEAD_TTC, I_LEAD_LATERAL_OFFSET]],
        [gap, -2.0, gap / 2.0, 0.5],
        atol=1e-6,
    )
    assert f.lead.id == 7


def test_lead_filters():
    ego = EgoState(0, 50, 0, 0, 10, 0, 0)
    oncoming = ObjectState(1, ObjectClass.CAR, 70, 3.5, 0, np.pi, 4.5, 1.9, 1.5, -10, 0)
    behind = ObjectState(2, ObjectClass.CAR, 40, 0, 0, 0, 4.5, 1.9, 1.5, 10, 0)
    ped_behind = ObjectState(3, ObjectClass.PEDESTRIAN, 45, 1, 0, 0, 0.5, 0.5, 1.8, 0, 0)
    f = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89)).extract(
        _obs(ego=ego, objects=(oncoming, behind, ped_behind)), Control(0, 0), 0.05
    )
    assert f.lead is None and f.raw[I_LEAD_DIST] == 80 and f.raw[I_PED_DIST] == 50


def test_dynamic_traffic_group():
    ped = ObjectState(4, ObjectClass.PEDESTRIAN, 60, 5, 0, -np.pi / 2, 0.5, 0.5, 1.8, 0, -1.2)
    cross = ObjectState(5, ObjectClass.CAR, 80, -30, 0, np.pi / 2, 4.5, 1.9, 1.5, 0, 10)
    f = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89)).extract(
        _obs(ego=EgoState(0, 50, 0, 0, 10, 0, 0), objects=(ped, cross), dist_to_junction=25.0),
        Control(0, 0),
        0.05,
    )
    np.testing.assert_allclose(
        f.raw[[I_PED_DIST, I_PED_BEARING]], [np.hypot(10, 5), np.arctan2(5, 10)], atol=1e-6
    )
    np.testing.assert_allclose(
        f.raw[[I_CROSS_VEH_DIST, I_CROSS_VEH_SPEED, I_CROSS_VEH_TTC]], [30, 10, 3.0], atol=1e-6
    )
    ego_clear = (80 - 51.7 + 4.7 + 2.0) / 10.0  # 3.5 s
    assert abs(f.raw[I_INTERSECTION_GAP] - (3.0 - ego_clear)) < 1e-6


def test_context_group():
    f = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89)).extract(
        _obs(ego=EgoState(0, 50, 0, 0, 10, 0, 0), light=L.RED, dist_to_stop_line=40.0), Control(0, 0), 0.05
    )
    np.testing.assert_allclose(
        f.raw[[I_TRAFFIC_LIGHT_STATE, I_SPEED_LIMIT, I_LANE_WIDTH, I_STOP_DIST]], [3, 13.89, 3.5, 40.0]
    )
    assert abs(f.stop_s - (51.7 + 40.0)) < 1e-6


def test_oncoming_on_curve_not_crossing():
    r = RouteBuilder.arc(30, np.pi / 2, 13.89)  # centre (0, 30), ego heads +x then curves left
    # oncoming car in the adjacent lane, driving back along the route: heading ~ -x (170 deg off the
    # tangent there), its straight path cuts the arc chord
    o = ObjectState(1, ObjectClass.CAR, 25.0, 6.0, 0, np.pi, 4.5, 1.9, 1.5, -10.0, 0)
    f = FeatureExtractor(VehicleParams(), r).extract(
        _obs(EgoState(0, 2, 0, 0, 10, 0, 0), [o]), Control(0, 0), 0.05
    )
    assert [f.raw[I_CROSS_VEH_DIST], f.raw[I_CROSS_VEH_SPEED], f.raw[I_CROSS_VEH_TTC]] == [80, 0, 10]


def test_absent_sentinels():
    f = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89)).extract(
        _obs(ego=EgoState(0, 50, 0, 0, 10, 0, 0)), Control(0, 0), 0.05
    )
    expect = {
        I_LEAD_DIST: 80,
        I_LEAD_REL_SPEED: 0,
        I_LEAD_TTC: 10,
        I_LEAD_ACCEL: 0,
        I_LEAD_LATERAL_OFFSET: 0,
        I_PED_DIST: 50,
        I_PED_BEARING: 0,
        I_CROSS_VEH_DIST: 80,
        I_CROSS_VEH_SPEED: 0,
        I_CROSS_VEH_TTC: 10,
        I_TRAFFIC_LIGHT_STATE: 0,
        I_INTERSECTION_GAP: 10,
        I_STOP_DIST: 100,
    }
    for i, v in expect.items():
        assert f.raw[i] == v, FEATURE_NAMES[i]
    assert f.stop_s is None and np.all(np.isfinite(f.raw))


def test_closing_speed_nonpositive_gives_ttc_hi():
    lead = ObjectState(7, ObjectClass.CAR, 70, 0, 0, 0, 4.5, 1.9, 1.5, 12.0, 0.0)  # pulling away
    f = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89)).extract(
        _obs(ego=EgoState(0, 50, 0, 0, 10, 0, 0), objects=(lead,)), Control(0, 0), 0.05
    )
    assert f.raw[I_LEAD_TTC] == 10 and f.raw[I_LEAD_REL_SPEED] == 2.0


def test_lead_accel_low_passed_and_reset_on_id_change():
    fx = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89))
    ego = EgoState(0, 50, 0, 0, 10, 0, 0)

    def lead(i, v):
        return ObjectState(i, ObjectClass.CAR, 70, 0, 0, 0, 4.5, 1.9, 1.5, v, 0.0)

    fx.extract(_obs(ego, (lead(1, 8.0),)), Control(0, 0), 0.05)
    a1 = fx.extract(_obs(ego, (lead(1, 7.9),)), Control(0, 0), 0.05).raw[I_LEAD_ACCEL]
    assert -2.0 < a1 < 0  # raw difference is -2 m/s^2, low-passed
    assert fx.extract(_obs(ego, (lead(2, 5.0),)), Control(0, 0), 0.05).raw[I_LEAD_ACCEL] == 0


def test_lateral_target_moves_corridor_and_reset_clears_it():
    fx = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89))
    obj = ObjectState(1, ObjectClass.CAR, 70, 3.0, 0, 0, 4.5, 1.9, 1.5, 8.0, 0.0)
    obs = _obs(EgoState(0, 50, 0, 0, 10, 0, 0), (obj,))
    assert fx.extract(obs, Control(0, 0), 0.05).lead is None
    fx.lateral_target = 3.0
    assert fx.extract(obs, Control(0, 0), 0.05).lead is not None
    fx.reset(RouteBuilder.straight(300, 13.89))
    assert fx.lateral_target == 0.0


def test_all_features_finite_for_random_observations():
    rng = np.random.default_rng(50_000)
    route = RouteBuilder.concat(RouteBuilder.straight(100, 13.89), RouteBuilder.arc(40, np.pi / 2, 13.89))
    fx = FeatureExtractor(VehicleParams(), route)
    for _ in range(300):
        ego = EgoState(0, rng.uniform(-20, 160), rng.uniform(-20, 60), rng.uniform(-4, 4), rng.uniform(0, 35),
                       rng.uniform(-4, 4), rng.uniform(-2, 2), rng.uniform(-8, 4))  # fmt: skip
        objs = tuple(
            ObjectState(
                i, ObjectClass(rng.integers(0, 5)), rng.uniform(-20, 200), rng.uniform(-30, 80), 0,
                rng.uniform(-4, 4), rng.uniform(0.3, 12), rng.uniform(0.3, 3), 1.5,
                rng.uniform(-25, 25), rng.uniform(-25, 25),
            )
            for i in range(int(rng.integers(0, 8)))
        )  # fmt: skip
        light = L(int(rng.integers(0, 4)))
        obs = _obs(
            ego,
            objs,
            light,
            rng.choice([np.inf, -5.0, 0.0, 30.0, 150.0]),
            rng.choice([np.inf, 0.0, 25.0, 90.0]),
        )
        f = fx.extract(obs, Control(rng.uniform(-0.7, 0.7), 0.0), 0.05)
        assert f.raw.shape == (24,) and np.all(np.isfinite(f.raw))
        assert np.all(np.abs(f.z) <= 1) and f.z.dtype == np.float32


def test_lead_beyond_route_end_keeps_true_gap():
    lead = ObjectState(7, ObjectClass.CAR, 130, 0.0, 0, 0, 4.5, 1.9, 1.5, 8.0, 0.0)  # 30 m past the end
    f = FeatureExtractor(VehicleParams(), RouteBuilder.straight(100, 13.89)).extract(
        _obs(ego=EgoState(0, 90, 0, 0, 10, 0, 0), objects=(lead,)), Control(0, 0), 0.05
    )
    np.testing.assert_allclose(f.raw[I_LEAD_DIST], 130 - 2.25 - 91.7, atol=1e-6)


def test_junction_range_speed_dependent():
    from hybrid_av.features.extractor import junction_range

    assert junction_range(0) == 50
    assert junction_range(12.7) == pytest.approx(12.7**2 / 2 + 10)


def test_intersection_gap_active_beyond_50m_at_speed():
    cross = ObjectState(5, ObjectClass.CAR, 80, -30, 0, np.pi / 2, 4.5, 1.9, 1.5, 0, 10)

    def gap(vx):
        fx = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89))
        o = _obs(ego=EgoState(0, 50, 0, 0, vx, 0, 0), objects=(cross,), dist_to_junction=80.0)
        return fx.extract(o, Control(0, 0), 0.05).raw[I_INTERSECTION_GAP]

    assert gap(12.7) < 10 and gap(5.0) == 10


def test_stop_dist_includes_junction_give_way_line_but_stop_s_does_not():
    def feats(vx, dj, **kw):
        fx = FeatureExtractor(VehicleParams(), RouteBuilder.straight(300, 13.89))
        return fx.extract(
            _obs(ego=EgoState(0, 50, 0, 0, vx, 0, 0), dist_to_junction=dj, **kw), Control(0, 0), 0.05
        )

    f = feats(5.0, 30.0)
    assert f.raw[I_STOP_DIST] == 30.0 and f.stop_s is None  # the reference never stops at a give-way line
    assert feats(5.0, 80.0).raw[I_STOP_DIST] == 100.0  # beyond junction_range(5.0) = 50
    assert feats(12.7, 80.0).raw[I_STOP_DIST] == 80.0  # junction_range(12.7) = 90.6
    assert feats(5.0, 30.0, light=L.RED, dist_to_stop_line=20.0).raw[I_STOP_DIST] == 20.0
