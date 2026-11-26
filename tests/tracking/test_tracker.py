import numpy as np
import pytest

from hybrid_av.perception.base import RadarDetection
from hybrid_av.tracking.radar import RadarExtrinsic
from hybrid_av.tracking.tracker import Tracker, TrackerConfig
from tests.tracking.synth import box_obs, ego_at, make_target, radar_obs

CFG = TrackerConfig()  # tracking.md §1 defaults
FRONT = RadarExtrinsic("front", 2.4, 0.0, 0.7, 0.0)
LEFT = RadarExtrinsic("left", 0.0, 1.0, 0.7, np.pi / 2)


def run(targets, frames, radars=(FRONT,), seed=0, box_sigma=0.3):
    rng, tr = np.random.default_rng(seed), Tracker(CFG, list(radars))
    for k in range(frames):
        t = k * 0.05
        e = ego_at(t, vx=0.0)
        tr.predict(t)
        tr.update_boxes(t, [box_obs(g, t, e, box_sigma, rng) for g in targets], e)
        tr.update_radar(
            t, [d for g in targets for ext in radars if (d := radar_obs(g, t, e, ext, 0.25, rng))], e
        )
    return tr


def vel_errors(target, frames, radars, seeds=range(20)):
    return np.array(
        [
            (tk.vx - target.vx, tk.vy - target.vy)
            for s in seeds
            for tk in run([target], frames, radars, seed=s).tracks()
        ]
    )


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def test_confirmation_needs_three_hits():
    tr = Tracker(CFG, [FRONT])
    g = make_target(20, 0, 0, 0)
    rng = np.random.default_rng(0)
    for k in range(2):
        tr.predict(k * 0.05)
        tr.update_boxes(k * 0.05, [box_obs(g, k * 0.05, ego_at(0), 0.0, rng)], ego_at(0))
    assert tr.tracks() == [] and len(tr.tracks(confirmed_only=False)) == 1
    tr.predict(0.1)
    tr.update_boxes(0.1, [box_obs(g, 0.1, ego_at(0), 0.0, rng)], ego_at(0))
    assert len(tr.tracks()) == 1


def test_tentative_track_deleted_after_two_misses():
    tr = Tracker(CFG, [FRONT])
    rng = np.random.default_rng(0)
    g = make_target(20, 0, 0, 0)
    tr.predict(0.0)
    tr.update_boxes(0.0, [box_obs(g, 0.0, ego_at(0), 0.0, rng)], ego_at(0))
    for t in (0.05, 0.10):
        tr.predict(t)
        tr.update_boxes(t, [], ego_at(t))
    assert tr.tracks(confirmed_only=False) == []


def test_tracks_age_out_without_detections():
    tr = run([make_target(20, 0, 0, 0)], 20)
    for k in range(5):
        t = 1.0 + k * 0.05
        tr.predict(t)
        tr.update_boxes(t, [], ego_at(t))
        tr.update_radar(t, [], ego_at(t))
    assert tr.tracks(confirmed_only=False) == []


def test_lidar_only_velocity_converges():
    # tracking.md §3: from a 6 m/s initial error to speed-error RMSE < 1.0 m/s after 40 frames (20 seeds).
    # Expected ≈ 0.6–0.85 m/s: the steady state for q_accel 2.0 and σ 0.3 m boxes at 20 Hz (simulated in review).
    e = vel_errors(make_target(25, 2, -6.0, 0.5), 40, radars=())
    assert len(e) == 20 and rms(np.linalg.norm(e, axis=1)) < 1.0


def test_front_radar_sharpens_radial_velocity():
    # target 5° off the front radar boresight (FOV ±15°): line of sight ≈ +x, so the radar observes vx.
    # Expected radial RMSE ≈ 0.15–0.25 m/s with radar vs ≈ 0.35–0.65 m/s LiDAR-only after 20 frames.
    g = make_target(25, 2, -6.0, 0.5)
    lid, rad = vel_errors(g, 20, ()), vel_errors(g, 20, (FRONT,))
    assert rms(rad[:, 0]) < 0.35 and rms(rad[:, 0]) < rms(lid[:, 0])


def test_two_crossing_targets_keep_ids():
    a, b = make_target(20, -6, 0, 3.0), make_target(20, 6, 0, -3.0)  # cross at t = 2 s
    tr = Tracker(CFG, [FRONT])
    rng = np.random.default_rng(1)
    ids_early = None
    for k in range(int(4.0 / 0.05)):
        t = k * 0.05
        e = ego_at(t)
        tr.predict(t)
        tr.update_boxes(t, [box_obs(a, t, e, 0.1, rng), box_obs(b, t, e, 0.1, rng)], e)
        tr.update_radar(t, [d for g in (a, b) if (d := radar_obs(g, t, e, FRONT, 0.1, rng))], e)
        if k == 20:
            ids_early = {round(tk.y): tk.id for tk in tr.tracks()}  # {-3: id_a, 3: id_b}
    late = {tk.id: tk.y for tk in tr.tracks()}
    assert late[ids_early[-3]] > 3 and late[ids_early[3]] < -3


def test_tracks_are_world_frame():
    tr = Tracker(CFG, [FRONT])
    rng = np.random.default_rng(0)
    g = make_target(20, 0, 0, 0)
    for k in range(4):
        t = k * 0.05
        e = ego_at(t, x0=50.0, psi=np.pi)  # ego at x=50 facing -x (30 m from the target, inside max_range)
        tr.predict(t)
        tr.update_boxes(t, [box_obs(g, t, e, 0.0, rng)], e)
    assert tr.tracks()[0].x == pytest.approx(20.0, abs=1e-6)


def test_radar_near_sensor_does_not_crash():
    tr = Tracker(CFG, [FRONT])
    g = make_target(FRONT.x, 0, 0, 0)  # confirmed track sitting on the radar
    rng = np.random.default_rng(0)
    for k in range(3):
        tr.predict(k * 0.05)
        tr.update_boxes(k * 0.05, [box_obs(g, k * 0.05, ego_at(0), 0.0, rng)], ego_at(0))
    before = tr.tracks()[0]
    tr.update_radar(0.1, [RadarDetection(1.0, 0.0, 0.0, -1.0, "front")], ego_at(0))
    after = tr.tracks()[0]
    assert (after.x, after.y, after.vx, after.vy) == (before.x, before.y, before.vx, before.vy)
    np.testing.assert_array_equal(after.pos_cov, before.pos_cov)
