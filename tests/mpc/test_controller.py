import numpy as np

from hybrid_av.mpc.controller import BicycleMPC, MPCConfig, Reference
from hybrid_av.types import EgoState
from hybrid_av.vehicle.dynamics import rk4_step
from hybrid_av.vehicle.params import VehicleParams
from tests.mpc.helpers import straight_reference

P, CFG = VehicleParams(), MPCConfig()


def _ego(**kw):
    d = {"t": 0.0, "x": 0.0, "y": 0.0, "psi": 0.0, "vx": 10.0, "vy": 0.0, "r": 0.0}
    d.update(kw)
    return EgoState(**d)


def test_on_reference_no_action():
    m = BicycleMPC(P, CFG)
    e = _ego()
    res = m.solve(e, straight_reference(e, 10.0), mu=1.0)
    assert res.status == "solved" and abs(res.control.delta) < 1e-3 and abs(res.control.ax) < 0.05


def test_lateral_error_steers_toward_path():
    m = BicycleMPC(P, CFG)
    e = _ego(y=-1.0)
    assert m.solve(e, straight_reference(e, 10.0), mu=1.0).control.delta > 0.0


def test_speed_error_and_brake_request():
    m = BicycleMPC(P, CFG)
    e = _ego(vx=8.0)
    assert m.solve(e, straight_reference(e, 10.0), mu=1.0).control.ax > 0.5
    m.reset()
    e = _ego()
    assert m.solve(e, straight_reference(e, 10.0), mu=1.0, brake_request=True).control.ax < -1.0


def test_heading_reference_aligned():
    m = BicycleMPC(P, CFG)
    e = _ego(psi=4 * np.pi)
    ref = straight_reference(_ego(), 10.0)  # ref.psi == 0, ego 2 laps ahead in heading
    assert abs(m.solve(e, ref, mu=1.0).control.delta) < 1e-3


def test_delay_compensation():
    m = BicycleMPC(P, CFG)
    e = _ego(y=-0.5)
    r1 = m.solve(e, straight_reference(e, 10.0), mu=1.0)
    r2 = m.solve(e, straight_reference(e, 10.0), mu=1.0)
    u1 = np.array([r1.control.delta, r1.control.ax])
    x_hat = rk4_step(e.as_mpc_vector(), u1, P, CFG.dt)
    assert np.linalg.norm(x_hat - rk4_step(e.as_mpc_vector(), np.zeros(2), P, CFG.dt)) > 1e-2  # u1 matters
    np.testing.assert_allclose(r2.predicted[0], x_hat, atol=2e-3)  # equality row, OSQP eps 1e-4 abs/rel


def test_warm_start_setup_once(monkeypatch):
    import osqp

    calls = {"setup": 0}
    orig = osqp.OSQP.setup

    def spy(self, *a, **k):
        calls["setup"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(osqp.OSQP, "setup", spy)
    m = BicycleMPC(P, CFG)
    e = _ego()
    for _ in range(20):
        m.solve(e, straight_reference(e, 10.0), mu=1.0)
    assert calls["setup"] == 1


def test_fallback_sequence(monkeypatch):
    m = BicycleMPC(P, CFG)
    e = _ego(y=-0.5)
    m.solve(e, straight_reference(e, 10.0), mu=1.0)
    monkeypatch.setattr(m, "_solve_qp", lambda *a, **k: None)  # None = solver failure
    s = [m.solve(e, straight_reference(e, 10.0), mu=1.0).status for _ in range(4)]
    assert s == ["fallback_shift", "fallback_shift", "fallback_brake", "fallback_brake"]
    last = m.solve(e, straight_reference(e, 10.0), mu=1.0)
    assert last.control.ax == CFG.ax_min and np.all(np.isfinite(last.predicted))


def test_solve_from_standstill():
    m = BicycleMPC(P, CFG)
    e = _ego(vx=0.0)
    res = m.solve(e, straight_reference(e, 5.0), mu=1.0)
    assert res.status in ("solved", "solved_inaccurate") and res.control.ax > 0.0


def test_nonpositive_mu_does_not_raise():
    for mu in (-1.0, 0.0):
        m = BicycleMPC(P, CFG)
        e = _ego()
        res = m.solve(e, straight_reference(e, 10.0), mu=mu)
        assert np.isfinite(res.control.delta) and np.isfinite(res.control.ax)
        assert res.status in ("solved", "solved_inaccurate", "fallback_shift", "fallback_brake")


def test_avoids_obstacle_on_path():
    from hybrid_av.geometry import obb_overlap
    from hybrid_av.types import ObjectClass, ObjectState

    # The 1 s horizon sees only 10 m at 10 m/s, less than the 11.1 m stopping distance at 4.5 m/s^2, so the MPC alone
    # cannot avoid a static obstacle (formulation section 1: the reference generator anticipates; the half-plane has a
    # fixed normal, no swerve). The reference therefore slows to a stop 6 m short of the obstacle, like the generator.
    obs = ObjectState(1, ObjectClass.CAR, 25.0, 0.0, 0.0, 0.0, 4.7, 1.9, 1.5, 0.0, 0.0)

    def ref_fn(e):
        xs, vs = np.empty(CFG.N + 1), np.empty(CFG.N + 1)
        xs[0], vs[0] = e.x + e.vx * CFG.dt, e.vx
        for k in range(1, CFG.N + 1):  # v^2 = 2 a d towards a stop point 6 m short of the obstacle
            vs[k] = min(10.0, np.sqrt(4.0 * max(obs.x - 6.0 - xs[k - 1], 0.0)))
            xs[k] = xs[k - 1] + vs[k] * CFG.dt
        z = np.zeros(CFG.N + 1)
        return Reference(xs, z, z.copy(), vs, z.copy(), xs.copy())

    m = BicycleMPC(P, CFG)
    x, u_apply, slack = np.array([10.0, 0, 0, 0, 0, 0]), np.zeros(2), 0.0
    for k in range(120):
        e = EgoState(k * CFG.dt, x[3], x[4], x[5], x[0], x[1], x[2])
        res = m.solve(e, ref_fn(e), mu=1.0, obstacles=[obs])
        x = rk4_step(x, u_apply, P, CFG.dt)  # one-step command delay
        u_apply = np.array([res.control.delta, res.control.ax])
        slack = max(slack, res.slack_obstacle_max)
        ego_box = (x[3], x[4], x[5], P.length, P.width)
        assert not obb_overlap(ego_box, (obs.x, obs.y, obs.yaw, obs.length, obs.width))
    assert slack < 1e-3 or x[0] < 0.5


def test_friction_limit_respected():
    from tests.mpc.helpers import circle_reference

    mu, R = 0.3, 15.0  # v = 10 m/s needs 6.7 m/s^2 > 0.85 * 0.3 * 9.81 = 2.5
    m = BicycleMPC(P, CFG)
    e = _ego(vx=10.0)
    res = m.solve(e, circle_reference(e, R, 10.0), mu=mu)
    lim = 0.85 * mu * 9.81
    lat = np.abs(res.predicted[1:, 0] * res.predicted[1:, 2])
    assert res.slack_friction_max > 1e-3 or np.all(lat <= lim + 1e-3)
    assert np.all(lat <= lim + 1e-3 + 10 * res.slack_friction_max)  # linearised rows: slack widens the band
    assert res.control.ax >= max(CFG.ax_min, -lim) - 1e-6


def test_cold_start_far_from_path_solves():
    m = BicycleMPC(P, CFG)
    e = _ego(vx=10.5037, y=1.8757, psi=0.4290)
    out = m.solve(e, straight_reference(e, 4.4423), 1.0)
    assert out.status in ("solved", "solved_inaccurate")
