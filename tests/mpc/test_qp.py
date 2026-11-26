import numpy as np

from hybrid_av.mpc.constraints import friction_rows, obstacle_halfplanes
from hybrid_av.mpc.controller import MPCConfig, Reference
from hybrid_av.mpc.linearize import linearize_horizon
from hybrid_av.mpc.qp import QPBuilder
from hybrid_av.vehicle.params import VehicleParams

P, CFG = VehicleParams(), MPCConfig()


def _inputs(v, r, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(21) * 0.05
    xb = np.column_stack(
        [np.full(21, v), np.zeros(21), np.full(21, r), v * t, rng.normal(size=21) * 0.1, np.zeros(21)]
    )
    ub = np.zeros((20, 2))
    A, B, c = linearize_horizon(xb, ub, P, 0.05)
    ref = Reference(x=v * t, y=np.zeros(21), psi=np.zeros(21), v=np.full(21, v), kappa=np.zeros(21), s=v * t)
    return (
        A,
        B,
        c,
        xb[0],
        ref,
        np.zeros(2),
        friction_rows(xb, 1.0, CFG),
        obstacle_halfplanes(xb, [], 4.7, 1.9, CFG, 0.05),
    )


def test_sizes():
    qb = QPBuilder(P, CFG)
    assert qb.n_var == 6 * 21 + 2 * 20 + 20 + 20 * 4 == 266
    assert qb.n_con == 6 + 120 + 40 + 20 + 40 + 80 + 100 == 406  # row blocks of mpc-formulation.md §9


def test_sparsity_constant():
    qb = QPBuilder(P, CFG)
    P1, _, A1, *_ = qb.build(*_inputs(10.0, 0.0, 0), brake_request=False)
    P2, _, A2, *_ = qb.build(*_inputs(3.0, 0.2, 1), brake_request=True)
    for M1, M2 in ((P1, P2), (A1, A2)):
        np.testing.assert_array_equal(M1.indptr, M2.indptr)
        np.testing.assert_array_equal(M1.indices, M2.indices)


def test_initial_state_and_input_bounds():
    qb = QPBuilder(P, CFG)
    args = _inputs(10.0, 0.0, 0)
    _, _, Acon, l, u = qb.build(*args, brake_request=False)
    assert Acon.shape == (qb.n_con, qb.n_var) and np.all(l <= u)
    # rows 0..5 pin x_0 exactly (first row block of mpc-formulation.md §9)
    np.testing.assert_array_equal(Acon[:6, :6].toarray(), np.eye(6))
    np.testing.assert_allclose(l[:6], args[3])
    np.testing.assert_allclose(u[:6], args[3])
    # input-bound rows (after 6 + 120): |delta| <= 0.70, ax in [-4.5, 2.5] at mu = 1
    np.testing.assert_allclose(u[126:166:2], 0.70)
    np.testing.assert_allclose(l[127:166:2], -4.5)


def _sym(M):
    M = M.toarray()
    return M + M.T - np.diag(np.diag(M))


def _hand_cost(z, qb, ref, u_prev):
    c, N = CFG, CFG.N
    X = z[: qb.iu0].reshape(N + 1, 6)
    U = z[qb.iu0 : qb.sf0].reshape(N, 2)
    tot = 0.0
    for k in range(N + 1):
        x = X[k]
        dx, dy = x[3] - ref.x[k], x[4] - ref.y[k]
        sp_, cp_ = np.sin(ref.psi[k]), np.cos(ref.psi[k])
        e_lat, e_lon, e_psi = -sp_ * dx + cp_ * dy, cp_ * dx + sp_ * dy, x[5] - ref.psi[k]
        ell = (
            c.Q[0] * (x[0] - ref.v[k]) ** 2
            + c.Q[1] * x[1] ** 2
            + c.Q[2] * (x[2] - ref.v[k] * ref.kappa[k]) ** 2
            + c.w_lat * e_lat**2
            + c.w_lon * e_lon**2
            + c.w_psi * e_psi**2
        )
        tot += (c.P_scale if k == N else 1.0) * ell
    prev = u_prev
    for k in range(N):
        tot += U[k] @ (np.array(c.R) * U[k]) + (U[k] - prev) @ (np.array(c.R_delta) * (U[k] - prev))
        prev = U[k]
    return tot


def test_qp_cost_matches_path_frame_cost():
    qb = QPBuilder(P, CFG)
    rng = np.random.default_rng(5)
    args = _inputs(8.0, 0.1, 2)
    ref = args[4]
    ref = Reference(
        x=ref.x, y=rng.normal(size=21), psi=rng.normal(size=21) * 0.3, v=ref.v,
        kappa=rng.normal(size=21) * 0.05, s=ref.s,
    )  # fmt: skip
    u_prev = np.array([0.03, -0.2])
    args = (*args[:4], ref, u_prev, *args[6:])
    Pm, q, *_ = qb.build(*args, brake_request=False)
    Ps = _sym(Pm)
    z1, z2 = (rng.normal(size=qb.n_var) for _ in range(2))
    for z in (z1, z2):
        z[qb.sf0 :] = 0.0
    qc = lambda z: 0.5 * z @ Ps @ z + q @ z
    got = qc(z1) - qc(z2)
    want = _hand_cost(z1, qb, ref, u_prev) - _hand_cost(z2, qb, ref, u_prev)
    assert abs(got - want) <= 1e-8 * max(abs(want), 1.0)


def test_dynamics_rows_hold_on_rollout():
    qb = QPBuilder(P, CFG)
    args = _inputs(10.0, 0.05, 3)
    A, B, c, x0 = args[:4]
    _, _, Acon, lo, up = qb.build(*args, brake_request=False)
    rng = np.random.default_rng(1)
    U = rng.normal(size=(20, 2)) * [0.05, 0.5]
    X = [x0]
    for k in range(20):
        X.append(A[k] @ X[k] + B[k] @ U[k] + c[k])
    z = np.zeros(qb.n_var)
    z[: qb.iu0] = np.ravel(X)
    z[qb.iu0 : qb.sf0] = U.ravel()
    n_eq = 6 + 6 * 20
    np.testing.assert_allclose((Acon @ z)[:n_eq], lo[:n_eq], atol=1e-9)
    np.testing.assert_array_equal(lo[:n_eq], up[:n_eq])
