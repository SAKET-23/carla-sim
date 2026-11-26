"""QP assembly for OSQP (mpc-formulation.md §9). The sparsity pattern is fixed; build() only rewrites values."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from hybrid_av.mpc.controller import MPCConfig, Reference
    from hybrid_av.vehicle.params import VehicleParams

NX, NU = 6, 2


class QPBuilder:
    def __init__(self, params: VehicleParams, cfg: MPCConfig) -> None:
        self.cfg, self.params = cfg, params
        N, J = cfg.N, cfg.max_obstacles
        self.iu0 = NX * (N + 1)
        self.sf0 = self.iu0 + NU * N
        self.so0 = self.sf0 + N
        self.n_var = self.so0 + N * J
        self.n_con = NX + NX * N + NU * N + N + 2 * N + N * J + N + N * J
        # Pattern: entries are stored even when a value is zero; vals = 1..nnz recovers the COO -> CSC order.
        self._P_perm, self._P_ptr, self._P_idx = self._pattern(self._p_entries(), (self.n_var,) * 2)
        z = (np.zeros((N, NX, NX)), np.zeros((N, NX, NU)), np.zeros((N, 6)), np.zeros((N, J, 3)))
        self._A_perm, self._A_ptr, self._A_idx = self._pattern(self._a_entries(*z), (self.n_con, self.n_var))

    @staticmethod
    def _pattern(entries, shape):
        rows, cols, vals = entries
        M = sp.coo_matrix((np.arange(1, len(vals) + 1.0), (rows, cols)), shape=shape).tocsc()
        assert M.nnz == len(vals)  # no duplicate coordinates
        return M.data.astype(int) - 1, M.indptr, M.indices

    def _csc(self, vals, perm, ptr, idx, shape):
        return sp.csc_matrix((vals[perm], idx, ptr), shape=shape)

    def _ix(self, k):
        return NX * np.asarray(k)

    def _iu(self, k):
        return self.iu0 + NU * np.asarray(k)

    def _p_entries(self, vals=None):
        """Row/col/value triples of the upper triangle of P (placeholder values unless `vals` given)."""
        N = self.cfg.N
        k = np.arange(N + 1)
        base = self._ix(k)[:, None]
        rr = base + np.array([0, 1, 2, 3, 3, 4, 5])
        cc = base + np.array([0, 1, 2, 3, 4, 4, 5])
        ku = np.arange(N)
        ur = (self._iu(ku)[:, None] + [0, 1]).ravel()
        ko = np.arange(N - 1)
        orr = (self._iu(ko)[:, None] + [0, 1]).ravel()
        occ = (self._iu(ko + 1)[:, None] + [0, 1]).ravel()
        rows = np.concatenate([rr.ravel(), ur, orr])
        cols = np.concatenate([cc.ravel(), ur, occ])
        return rows, cols, np.ones(len(rows)) if vals is None else vals

    def _a_entries(self, A, B, coef, H):
        N, J = self.cfg.N, self.cfg.max_obstacles
        ii, jj = (m.ravel() for m in np.meshgrid(range(NX), range(NX), indexing="ij"))
        iB, jB = (m.ravel() for m in np.meshgrid(range(NX), range(NU), indexing="ij"))
        k = np.arange(N)
        r_dyn = (NX + NX * k)[:, None]
        R, C, V = [np.arange(NX)], [np.arange(NX)], [np.ones(NX)]
        for rows, cols, vals in (
            (r_dyn + ii, self._ix(k)[:, None] + jj, -A.reshape(N, -1)),
            (r_dyn + np.arange(NX), self._ix(k + 1)[:, None] + np.arange(NX), np.ones((N, NX))),
            (r_dyn + iB, self._iu(k)[:, None] + jB, -B.reshape(N, -1)),
        ):
            R.append(rows.ravel()), C.append(cols.ravel()), V.append(vals.ravel())
        r0 = NX + NX * N
        R.append(r0 + np.arange(NU * N)), C.append(self.iu0 + np.arange(NU * N)), V.append(np.ones(NU * N))
        r0 += NU * N  # steering slew
        R += [r0 + k, r0 + k[1:]]
        C += [self._iu(k), self._iu(k[:-1])]
        V += [np.ones(N), -np.ones(N - 1)]
        r0 += N  # friction: two rows per step, shared slack
        for off, sgn in ((0, -1.0), (1, 1.0)):
            row = r0 + 2 * k + off
            R += [row, row, row]
            C += [self._ix(k + 1) + 0, self._ix(k + 1) + 2, self.sf0 + k]
            V += [coef[:, 0], coef[:, 2], np.full(N, sgn)]
        r0 += 2 * N
        kj = (k[:, None] * J + np.arange(J)).ravel()  # obstacle rows ordered (k, slot)
        row = r0 + kj
        xk = np.repeat(self._ix(k + 1), J)
        R += [row, row, row]
        C += [xk + 3, xk + 4, self.so0 + kj]
        V += [H[:, :, 0].ravel(), H[:, :, 1].ravel(), -np.ones(N * J)]
        r0 += N * J
        (
            R.append(r0 + np.arange(N + N * J)),
            C.append(self.sf0 + np.arange(N + N * J)),
            V.append(np.ones(N + N * J)),
        )
        return np.concatenate(R), np.concatenate(C), np.concatenate(V)

    def build(self, A, B, c, x0, ref: Reference, u_prev, fric, H, brake_request):
        cfg = self.cfg
        N, J = cfg.N, cfg.max_obstacles
        coef, off, bound = fric
        # ---- cost
        w = np.ones(N + 1)
        w[N] = cfg.P_scale
        psi = ref.psi
        t = np.column_stack([np.cos(psi), np.sin(psi)])
        n = np.column_stack([-np.sin(psi), np.cos(psi)])
        pref = np.column_stack([ref.x, ref.y])
        qvx = cfg.Q[0] * (10.0 if brake_request else 1.0)
        vref = np.zeros(N + 1) if brake_request else ref.v
        Hxx = cfg.w_lat * n[:, 0] ** 2 + cfg.w_lon * t[:, 0] ** 2
        Hxy = cfg.w_lat * n[:, 0] * n[:, 1] + cfg.w_lon * t[:, 0] * t[:, 1]
        Hyy = cfg.w_lat * n[:, 1] ** 2 + cfg.w_lon * t[:, 1] ** 2
        stage = np.column_stack(
            [
                np.full(N + 1, qvx),
                np.full(N + 1, cfg.Q[1]),
                np.full(N + 1, cfg.Q[2]),
                Hxx,
                Hxy,
                Hyy,
                np.full(N + 1, cfg.w_psi),
            ]
        )
        pv_x = 2 * w[:, None] * stage
        Rd = np.array(cfg.R_delta)
        mult = np.where(np.arange(N) < N - 1, 2.0, 1.0)[:, None]
        pv_u = (2 * (np.array(cfg.R) + Rd * mult)).ravel()
        pv_o = np.tile(-2 * Rd, N - 1)
        Pval = np.concatenate([pv_x.ravel(), pv_u, pv_o])
        q = np.zeros(self.n_var)
        qx = np.zeros((N + 1, NX))
        qx[:, 0] = -2 * w * qvx * vref
        qx[:, 2] = -2 * w * cfg.Q[2] * ref.v * ref.kappa
        qx[:, 3:5] = (
            -2
            * w[:, None]
            * (
                cfg.w_lat * n * np.sum(n * pref, 1, keepdims=True)
                + cfg.w_lon * t * np.sum(t * pref, 1, keepdims=True)
            )
        )
        qx[:, 5] = -2 * w * cfg.w_psi * psi
        q[: self.iu0] = qx.ravel()
        q[self.iu0 : self.iu0 + NU] = -2 * Rd * u_prev
        q[self.sf0 : self.so0] = cfg.w_slack_friction
        q[self.so0 :] = cfg.w_slack_obstacle
        Pm = self._csc(Pval, self._P_perm, self._P_ptr, self._P_idx, (self.n_var,) * 2)
        # ---- constraints
        Acon = self._csc(
            self._a_entries(A, B, coef, H)[2],
            self._A_perm,
            self._A_ptr,
            self._A_idx,
            (self.n_con, self.n_var),
        )
        ax_b = bound[0]
        lo = np.tile([-cfg.delta_max, max(cfg.ax_min, -ax_b)], N)
        hi = np.tile([cfg.delta_max, min(cfg.ax_max, ax_b)], N)
        slew_l, slew_u = np.full(N, -cfg.ddelta_max), np.full(N, cfg.ddelta_max)
        slew_l[0] += u_prev[0]
        slew_u[0] += u_prev[0]
        inf = np.inf
        zN = np.zeros(N + N * J)
        l = np.concatenate(
            [
                x0,
                c.ravel(),
                lo,
                slew_l,
                np.column_stack([np.full(N, -inf), -bound - off]).ravel(),
                np.full(N * J, -inf),
                zN,
            ]
        )
        u = np.concatenate(
            [
                x0,
                c.ravel(),
                hi,
                slew_u,
                np.column_stack([bound - off, np.full(N, inf)]).ravel(),
                H[:, :, 2].ravel(),
                np.full(N + N * J, inf),
            ]
        )
        return Pm, q, Acon, l, u
