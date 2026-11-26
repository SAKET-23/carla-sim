import numpy as np
import pytest

from hybrid_av.tracking.ekf import cv_predict, radar_h, radar_jacobian, update_position, update_radar


def test_predict_moves_with_velocity():
    x, P = cv_predict(np.array([0, 0, 2.0, -1.0]), np.eye(4), dt=0.5, q_accel=1.0)
    np.testing.assert_allclose(x, [1.0, -0.5, 2.0, -1.0])
    assert np.allclose(P, P.T) and P[0, 0] > 1.0


def test_radar_h_value():
    assert radar_h(np.array([10.0, 0, -3.0, 5.0]), np.array([0.0, 0.0])) == pytest.approx(-3.0)


def test_radar_jacobian_matches_finite_difference():
    # tracking-fusion.md §4: 100 random states, central differences step 1e-6, max relative error < 1e-5
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = rng.normal(size=4) * [20, 20, 5, 5] + [25, 0, 0, 0]
        s = rng.normal(size=2)
        J = radar_jacobian(x, s).ravel()
        num = np.array([(radar_h(x + e, s) - radar_h(x - e, s)) / 2e-6 for e in np.eye(4) * 1e-6])
        assert np.max(np.abs(J - num) / np.maximum(np.abs(num), 1e-3)) < 1e-5


def test_position_update_pulls_toward_measurement_and_reports_nis():
    x, P, nis = update_position(np.zeros(4), np.eye(4), np.array([1.0, 0.0]), np.eye(2))
    assert x[0] == pytest.approx(0.5) and nis == pytest.approx(0.5)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_radar_update_corrects_radial_velocity_only():
    x0 = np.array([20.0, 0.0, 0.0, 0.0])
    P0 = np.diag([0.1, 0.1, 4.0, 4.0])
    x, _, _ = update_radar(x0, P0, rr=-5.0, sensor_xy=np.zeros(2), r_var=0.01)
    assert x[2] == pytest.approx(-5.0, abs=0.05)  # radial = x direction
    assert x[3] == pytest.approx(0.0, abs=1e-9)  # tangential unobserved


def test_radar_inside_vehicle_raises():
    with pytest.raises(ValueError):
        radar_h(np.array([0.1, 0.0, 1.0, 0.0]), np.zeros(2))
