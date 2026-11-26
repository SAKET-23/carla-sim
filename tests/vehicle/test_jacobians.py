import numpy as np

from hybrid_av.vehicle.dynamics import f_blended, jacobians
from hybrid_av.vehicle.params import VehicleParams

P = VehicleParams()


def _fd(x, u, h=1e-6):
    A = np.zeros((6, 6))
    B = np.zeros((6, 2))
    for i in range(6):
        e = np.zeros(6)
        e[i] = h
        A[:, i] = (f_blended(x + e, u, P) - f_blended(x - e, u, P)) / (2 * h)
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        B[:, j] = (f_blended(x, u + e, P) - f_blended(x, u - e, P)) / (2 * h)
    return A, B


def test_jacobians_match_finite_differences():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = np.array(
            [
                rng.uniform(0.5, 25),
                rng.uniform(-1, 1),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-50, 50),
                rng.uniform(-50, 50),
                rng.uniform(-7, 7),
            ]
        )
        if abs(x[0] - 3.0) < 1e-3 or abs(x[0] - 5.0) < 1e-3:  # blend kinks: not differentiable
            continue
        u = np.array([rng.uniform(-0.7, 0.7), rng.uniform(-4.5, 2.5)])
        A, B = jacobians(x, u, P)
        Af, Bf = _fd(x, u)
        np.testing.assert_allclose(A, Af, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(B, Bf, rtol=1e-5, atol=1e-6)


def test_jacobians_match_fd_at_fixed_speeds():
    # includes the 3-5 m/s blend ramp (4.0) and the V_EPS clamp region (0.1)
    for vx in (0.0, 0.1, 1.0, 4.0, 10.0):
        x = np.array([vx, 0.3, 0.2, 1.0, -2.0, 0.7])
        u = np.array([0.25, 1.0])
        A, B = jacobians(x, u, P)
        Af, Bf = _fd(x, u)
        np.testing.assert_allclose(A, Af, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(B, Bf, rtol=1e-5, atol=1e-6)


def test_vy_r_coupling_present():
    A, _ = jacobians(np.array([10.0, 0.3, 0.2, 0, 0, 0]), np.zeros(2), P)
    assert np.isclose(A[0, 2], 0.3) and np.isclose(A[0, 1], 0.2)


def test_jacobians_finite_at_zero_speed():
    A, B = jacobians(np.array([0.0, 0, 0, 0, 0, 0]), np.array([0.3, 1.0]), P)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(B))
