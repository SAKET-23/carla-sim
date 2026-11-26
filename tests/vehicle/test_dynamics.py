import numpy as np

from hybrid_av.vehicle.dynamics import f_blended, f_dynamic, f_kinematic
from hybrid_av.vehicle.params import VehicleParams

P = VehicleParams()


def test_straight_line():
    np.testing.assert_allclose(
        f_dynamic(np.array([10.0, 0, 0, 0, 0, 0]), np.zeros(2), P), [0, 0, 0, 10, 0, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        f_dynamic(np.array([10.0, 0, 0, 0, 0, np.pi / 2]), np.zeros(2), P)[3:5], [0, 10], atol=1e-12
    )


def test_dynamic_steer_response():  # original plan's equations, Cf for the front axle
    xd = f_dynamic(np.array([10.0, 0, 0, 0, 0, 0]), np.array([0.05, 0.0]), P)
    assert np.isclose(xd[1], 19000 * 0.05 * np.cos(0.05) / 1500, rtol=1e-9)  # 0.63254 m/s²
    assert np.isclose(xd[2], 1.2 * 19000 * 0.05 * np.cos(0.05) / 2250, rtol=1e-9)  # 0.50603 rad/s²


def test_kinematic_relaxes_to_kinematic_yaw_rate():  # tau_k = 0.1 s (vehicle-model.md §4)
    vx, d = 2.0, 0.1
    r_kin = vx * np.tan(d) / P.L
    xd = f_kinematic(np.array([vx, 0, 0, 0, 0, 0]), np.array([d, 0.0]), P)
    assert np.isclose(xd[2], r_kin / 0.1, rtol=1e-9) and np.isclose(xd[1], P.b * r_kin / 0.1, rtol=1e-9)
    xd = f_kinematic(np.array([vx, P.b * r_kin, r_kin, 0, 0, 0]), np.array([d, 0.0]), P)
    np.testing.assert_allclose(xd[[1, 2]], 0.0, atol=1e-12)
    assert np.isclose(xd[5], r_kin, rtol=1e-12) and np.isclose(xd[0], 0.0)


def test_blend_endpoints_and_continuity():
    u = np.array([0.1, 0.5])
    for vx, ref in ((2.0, f_kinematic), (3.0, f_kinematic), (5.0, f_dynamic), (8.0, f_dynamic)):
        x = np.array([vx, 0.05, 0.02, 0, 0, 0.3])
        np.testing.assert_allclose(f_blended(x, u, P), ref(x, u, P), atol=1e-12)
    a = f_blended(np.array([4.0 - 1e-7, 0.05, 0.02, 0, 0, 0.3]), u, P)
    b = f_blended(np.array([4.0 + 1e-7, 0.05, 0.02, 0, 0, 0.3]), u, P)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_low_speed_finite():
    for vx in (0.0, 0.1, 1.0):
        assert np.all(np.isfinite(f_blended(np.array([vx, 0.2, 0.3, 0, 0, 0]), np.array([0.7, -4.5]), P)))
