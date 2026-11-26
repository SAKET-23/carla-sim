import numpy as np

from hybrid_av.eval.metrics import (
    cross_track_rmse,
    heading_rmse_deg,
    max_lat_accel_ratio,
    solve_time_percentiles,
    steering_jerk_rms,
)


def test_steering_jerk_sine():
    t = np.arange(0, 20, 0.05)
    d = 0.1 * np.sin(2 * t)
    assert abs(steering_jerk_rms(d, 0.05) - 0.8 / np.sqrt(2)) / (0.8 / np.sqrt(2)) < 0.02


def test_simple_metrics():
    assert cross_track_rmse(np.array([3.0, -4.0])) == np.sqrt(12.5)
    assert abs(heading_rmse_deg(np.radians(np.array([2.0, -2.0]))) - 2.0) < 1e-12
    assert max_lat_accel_ratio(np.array([10.0, 20.0]), np.array([0.1, -0.2]), np.array([1.0, 0.5])) == 4.0 / (
        0.5 * 9.81
    )
    p = solve_time_percentiles(np.arange(1.0, 101.0))
    assert set(p) == {"p50", "p99"} and abs(p["p50"] - 50.5) < 1e-9
