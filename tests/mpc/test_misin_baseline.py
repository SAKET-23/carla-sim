import numpy as np

from hybrid_av.mpc.controller import BicycleMPC, MPCConfig
from hybrid_av.vehicle.params import VehicleParams
from tests.mpc.helpers import circle_reference, closed_loop, straight_reference
from tests.mpc.misin_baseline import MisinLPVMPC, lpv_matrices

P, CFG = VehicleParams(), MPCConfig()


def test_lpv_exact_at_scheduling_point():
    from hybrid_av.vehicle.dynamics import f_dynamic

    x = np.array([12.0, 0.3, 0.1, 5.0, -2.0, 0.7])
    u = np.array([0.05, 0.5])
    xi, r_ref = np.array([12.0, 0.3, 0.1, 0.4, 0.2]), 0.05  # [vx, vy, r, e_lat, e_psi]
    A, B, E = lpv_matrices(xi, P, delta=u[0])
    xi_dot = A @ xi + B @ u + E * r_ref
    np.testing.assert_allclose(xi_dot[:3], f_dynamic(x, u, P)[:3], atol=1e-9)
    assert np.isclose(xi_dot[3], 12.0 * np.sin(0.2) + 0.3 * np.cos(0.2), atol=1e-9)
    assert np.isclose(xi_dot[4], 0.1 - r_ref, atol=1e-12)


def _rmse_straight(ctrl):
    out = closed_loop(
        ctrl, lambda e: straight_reference(e, 10.0), np.array([10.0, 0, 0, 0, 0.3, 0]), steps=400
    )
    return np.sqrt(np.mean(out["x"][40:, 4] ** 2))


def _rmse_circle(ctrl, R=30.0, v=8.0):
    steps = int(2 * np.pi * R / v / 0.05)
    out = closed_loop(ctrl, lambda e: circle_reference(e, R, v), np.array([v, 0, 0, 0, 0, 0]), steps)
    X, Y = out["x"][:, 3], out["x"][:, 4]
    return np.sqrt(np.mean((np.hypot(X, Y - R) - R) ** 2))


def test_baseline_runs_and_tracks():
    assert _rmse_straight(MisinLPVMPC(P, CFG)) < 0.5
    assert _rmse_circle(MisinLPVMPC(P, CFG)) < 1.0


def test_plan_mpc_at_least_as_good_as_baseline():
    s_plan, s_base = _rmse_straight(BicycleMPC(P, CFG)), _rmse_straight(MisinLPVMPC(P, CFG))
    c_plan, c_base = _rmse_circle(BicycleMPC(P, CFG)), _rmse_circle(MisinLPVMPC(P, CFG))
    print(
        f"straight RMSE plan {s_plan:.3f} baseline {s_base:.3f}; circle RMSE plan {c_plan:.3f} baseline {c_base:.3f}"
    )
    assert s_plan <= s_base + 0.02 and c_plan <= c_base + 0.02
