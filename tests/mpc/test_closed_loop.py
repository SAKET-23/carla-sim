# ruff: noqa: B023  (loop lambdas are called within their own iteration)
import numpy as np
import pytest

from hybrid_av.geometry import wrap_angle
from hybrid_av.mpc.controller import BicycleMPC, MPCConfig
from hybrid_av.vehicle.params import VehicleParams
from tests.mpc.helpers import circle_reference, closed_loop, straight_reference

P, CFG = VehicleParams(), MPCConfig()


def test_straight_tracking_rmse():
    out = closed_loop(
        BicycleMPC(P, CFG),
        lambda e: straight_reference(e, 10.0),
        np.array([10.0, 0, 0, 0, 0.3, 0]),
        steps=400,
    )
    y = out["x"][:, 4]
    assert np.sqrt(np.mean(y[40:] ** 2)) < 0.10  # after 2 s
    assert np.max(np.abs(y)) < 0.35


def test_circle_two_laps():
    R, v = 30.0, 8.0
    steps = int(2 * 2 * np.pi * R / v / 0.05) + 20
    out = closed_loop(
        BicycleMPC(P, CFG), lambda e: circle_reference(e, R, v), np.array([v, 0, 0, 0, 0, 0]), steps
    )
    X, Y, psi = out["x"][:, 3], out["x"][:, 4], out["x"][:, 5]
    assert psi[-1] > 4 * np.pi - 0.5  # two full laps, unwrapped
    psi_ref = np.arctan2(X, R - Y)  # tangent of the CCW circle about (0, R)
    assert np.max(np.abs(wrap_angle(psi - psi_ref))) < 0.5
    assert np.sqrt(np.mean((np.hypot(X, Y - R) - R) ** 2)) < 0.3


def _check_random_references(n, seed):
    rng = np.random.default_rng(seed)
    for i in range(n):
        m = BicycleMPC(P, CFG)
        v = rng.uniform(2, 25)
        x0 = np.array([v, 0, 0, 0, rng.uniform(-2, 2), rng.uniform(-0.5, 0.5)])
        v_ref, radius = rng.uniform(0, 25), rng.uniform(20, 150)  # drawn once per case, not per step
        ref = (
            (lambda e: straight_reference(e, v_ref)) if i % 2 else (lambda e: circle_reference(e, radius, v))
        )
        out = closed_loop(m, ref, x0, steps=20)
        u = out["u"]
        assert np.all(np.abs(u[:, 0]) <= 0.70 + 1e-3)
        assert np.all(np.abs(np.diff(np.vstack([[0.0, 0.0], u])[:, 0])) <= 0.15 + 1e-3)
        assert np.all((u[:, 1] >= -4.5 - 1e-3) & (u[:, 1] <= 2.5 + 1e-3))
        assert "fallback_brake" not in list(out["status"])


@pytest.mark.slow
def test_constraints_random_references():
    _check_random_references(200, 3)


def test_constraints_random_references_fast():
    _check_random_references(40, 4)
