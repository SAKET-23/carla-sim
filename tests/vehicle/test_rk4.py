import numpy as np

from hybrid_av.vehicle.dynamics import f_blended, rk4_step
from hybrid_av.vehicle.params import VehicleParams

P = VehicleParams()


def test_straight_exact():
    np.testing.assert_allclose(
        rk4_step(np.array([10.0, 0, 0, 0, 0, 0]), np.zeros(2), P, 0.05),
        [10, 0, 0, 0.5, 0, 0],
        atol=1e-12,
    )


def test_matches_fine_euler():
    x0 = np.array([12.0, 0.2, 0.1, 0, 0, 0.4])
    u = np.array([0.08, -1.0])
    x = x0.copy()
    for _ in range(5000):
        x = x + 1e-5 * f_blended(x, u, P)
    np.testing.assert_allclose(rk4_step(x0, u, P, 0.05), x, atol=1e-4)


def test_standstill_start_finite_and_kinematic_radius():  # vehicle-model.md §4 pinned test
    x = np.zeros(6)
    for k in range(200):  # 10 s: 2 m/s² for 1 s, then hold vx = 2 m/s
        x = rk4_step(x, np.array([0.3, 2.0 if k < 20 else 0.0]), P, 0.05)
        assert np.all(np.isfinite(x))
    assert np.isclose(x[0], 2.0, rtol=1e-9) and np.isclose(x[0] / x[2], P.L / np.tan(0.3), rtol=1e-3)
