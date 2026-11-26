import numpy as np

from hybrid_av.mpc.linearize import linearize, linearize_horizon
from hybrid_av.vehicle.dynamics import rk4_step
from hybrid_av.vehicle.params import VehicleParams

P, DT = VehicleParams(), 0.05
XB, UB = np.array([12.0, 0.3, 0.15, 5, -3, 2.0]), np.array([0.1, 1.0])


def test_exact_at_linearisation_point():
    Ad, Bd, cd = linearize(XB, UB, P, DT)
    np.testing.assert_allclose(Ad @ XB + Bd @ UB + cd, rk4_step(XB, UB, P, DT), atol=1e-9)


def test_accurate_nearby():
    Ad, Bd, cd = linearize(XB, UB, P, DT)
    rng = np.random.default_rng(1)
    for _ in range(10):
        dx = rng.normal(size=6) * 1e-3
        du = rng.normal(size=2) * 1e-3
        err = Ad @ (XB + dx) + Bd @ (UB + du) + cd - rk4_step(XB + dx, UB + du, P, DT)
        assert np.linalg.norm(err) < 5e-5


def test_horizon_shapes():
    A, B, c = linearize_horizon(np.tile(XB, (21, 1)), np.tile(UB, (20, 1)), P, DT)
    assert A.shape == (20, 6, 6) and B.shape == (20, 6, 2) and c.shape == (20, 6)
