"""Mahalanobis gating and Hungarian assignment (tracking-fusion.md §5)."""

import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG = 1e6


def mahalanobis_cost(track_pos, track_cov, det_pos, det_cov) -> np.ndarray:
    """(T, D) squared Mahalanobis distances with S = track_cov[t] + det_cov[d]."""
    nu = np.asarray(det_pos)[None, :, :] - np.asarray(track_pos)[:, None, :]  # (T, D, 2)
    S = np.asarray(track_cov)[:, None] + np.asarray(det_cov)[None, :]  # (T, D, 2, 2)
    return np.einsum("tdi,tdi->td", nu, np.linalg.solve(S, nu[..., None])[..., 0])


def assign(cost: np.ndarray, gate: float = 9.21):
    """Returns (matches, unmatched_tracks, unmatched_dets); pairs above the gate never match."""
    T, D = cost.shape
    matches = []
    if T and D:
        rows, cols = linear_sum_assignment(np.where(cost > gate, _BIG, cost))
        matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= gate]
    mt, md = {r for r, _ in matches}, {c for _, c in matches}
    return matches, [i for i in range(T) if i not in mt], [j for j in range(D) if j not in md]
