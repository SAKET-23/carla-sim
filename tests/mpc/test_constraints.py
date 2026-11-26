import numpy as np

from hybrid_av.mpc.constraints import G, friction_rows, obstacle_halfplanes
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.types import ObjectClass, ObjectState

CFG = MPCConfig()


def _xbar(v=10.0, r=0.1):
    t = np.arange(21) * 0.05
    return np.column_stack([np.full(21, v), np.zeros(21), np.full(21, r), v * t, np.zeros(21), np.zeros(21)])


def test_friction_rows_exact_at_point():
    xb = _xbar()
    coef, off, bound = friction_rows(xb, 0.35, CFG)
    np.testing.assert_allclose(np.einsum("kj,kj->k", coef, xb[1:]) + off, 10.0 * 0.1)
    np.testing.assert_allclose(bound, 0.85 * 0.35 * G)


def test_halfplane_separates():
    xb = _xbar(r=0.0)
    obs = ObjectState(
        id=1,
        cls=ObjectClass.CAR,
        x=20.0,
        y=0.0,
        z=0.0,
        yaw=0.0,
        length=4.5,
        width=1.9,
        height=1.5,
        vx=0.0,
        vy=0.0,
    )
    H = obstacle_halfplanes(xb, [obs], ego_length=4.7, ego_width=1.9, cfg=CFG, dt=0.05)
    assert H.shape == (20, 4, 3)
    for k in range(20):
        a, b, c = H[k, 0]
        assert a * 20.0 + b * 0.0 > c  # obstacle centre on the forbidden side
        assert a * xb[k + 1, 3] + b * xb[k + 1, 4] <= c + 1e-9  # ego reaches at most X = 10 m
        assert np.isclose(a * 20.0 - c, 2.25 + 2.35 + 1.0)  # obstacle half-length + ego half-length + margin
    np.testing.assert_array_equal(H[:, 1:], np.tile([0.0, 0.0, 1.0], (20, 3, 1)))


def test_halfplane_degenerate_normal():
    xb = _xbar(v=0.0, r=0.0)
    obs = ObjectState(
        id=1,
        cls=ObjectClass.CAR,
        x=0.0,
        y=0.0,
        z=0.0,
        yaw=0.3,
        length=4.5,
        width=1.9,
        height=1.5,
        vx=0.0,
        vy=0.0,
    )
    H = obstacle_halfplanes(xb, [obs], ego_length=4.7, ego_width=1.9, cfg=CFG, dt=0.05)
    assert np.all(np.isfinite(H)) and np.allclose(np.hypot(H[:, 0, 0], H[:, 0, 1]), 1.0)
