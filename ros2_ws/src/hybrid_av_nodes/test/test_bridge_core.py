import numpy as np
import pytest
from hybrid_av_nodes import bridge_core as B

from hybrid_av.perception.base import RadarDetection
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.tracking.radar import RadarExtrinsic
from hybrid_av.types import ActuatorCommand
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams


def test_bgra_to_rgb():
    raw = bytes([10, 20, 30, 255] * 6)  # B, G, R, A
    rgb = B.image_bgra_to_rgb(raw, 3, 2)
    assert rgb.shape == (2, 3, 3) and rgb[0, 0].tolist() == [30, 20, 10]


def test_semantic_tag_from_red_channel():
    raw = bytes([0, 0, 14, 255, 0, 0, 1, 255])
    assert B.semantic_tags(raw, 2, 1).tolist() == [[14, 1]]


def test_camera_info_fov90():
    ci = B.camera_info(1280, 720, 90.0)
    assert ci["k"][0] == pytest.approx(640.0) and ci["k"][2] == pytest.approx(640.0)
    assert (
        ci["k"][5] == pytest.approx(360.0) and ci["d"] == [0.0] * 5 and ci["distortion_model"] == "plumb_bob"
    )
    assert ci["p"][0] == pytest.approx(640.0) and ci["p"][10] == 1.0


def test_radar_points_layout():
    ext = RadarExtrinsic("front", 2.4, 0.0, 0.7, 0.0)
    p = B.radar_points([RadarDetection(10.0, 0.0, 0.0, -2.0, "front")], ext)
    assert p.shape == (1, 7) and p[0, :3].tolist() == pytest.approx([12.4, 0.0, 0.7]) and p[0, 3] == -2.0
    assert B.radar_points([], ext).shape == (0, 7)


def test_imu_stamps_interpolated():
    assert B.imu_stamps(1.0, 5) == pytest.approx([0.96, 0.97, 0.98, 0.99, 1.0])
    assert B.imu_stamps(1.0, 1) == [1.0]


def test_wheel_odom_seeded_and_biased():
    a, b = B.WheelOdomSynth(seed=3), B.WheelOdomSynth(seed=3)
    xs = [a(10.0, 0.0)[0] for _ in range(2000)]
    assert xs[:5] == [b(10.0, 0.0)[0] for _ in range(5)]
    assert np.mean(xs) == pytest.approx(10.1, abs=0.01) and np.std(xs) == pytest.approx(0.05, rel=0.1)


def test_episode_driver_publishes_reset_observation_first():
    """Tick 0 returns the reset observation without stepping (same as litesim_node)."""
    sim = LiteSim(VehicleParams(), LongitudinalMap())
    d = B.EpisodeDriver(sim, make_scenario("nominal", 0))
    t0, done0 = d.step(ActuatorCommand(0.0, 0.5, 0.0))
    assert t0 == 0.0 and not done0 and d.info is None
    t1, _ = d.step(ActuatorCommand(0.0, 0.5, 0.0))
    assert t1 == pytest.approx(sim.dt) and d.info is not None


def test_episode_driver_summary_fields():
    sim = LiteSim(VehicleParams(), LongitudinalMap())
    d = B.EpisodeDriver(sim, make_scenario("nominal", 0))
    for _ in range(5):
        d.step(ActuatorCommand(0.0, 0.5, 0.0))
    s = d.summary("nominal", 0)
    assert {"scenario", "seed", "route_completion", "infractions", "driving_score", "duration"} <= set(s)
    assert 0.0 <= s["route_completion"] <= 1.0
