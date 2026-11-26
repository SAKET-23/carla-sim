"""Absolute trajectory error and relative pose error (metrics.md P1)."""

import numpy as np


def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """(R, t, s) minimising ||dst - (s R src + t)||; s is exactly 1.0 without scale."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.ones(src.shape[1])
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0  # no reflections
    R = u @ np.diag(d) @ vt
    s = float((sig * d).sum() / xs.var(0).sum()) if with_scale else 1.0
    return R, mu_d - s * R @ mu_s, s


def associate(t_est: np.ndarray, t_gt: np.ndarray, max_dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs by nearest timestamp (t_gt sorted). shortcut: several estimates may share one GT sample."""
    j = np.clip(np.searchsorted(t_gt, t_est), 1, len(t_gt) - 1)
    j = np.where(np.abs(t_gt[j - 1] - t_est) <= np.abs(t_gt[j] - t_est), j - 1, j)
    i = np.flatnonzero(np.abs(t_gt[j] - t_est) <= max_dt)
    return i, j[i]


def ate_rmse(est: np.ndarray, gt: np.ndarray, align: bool = True) -> float:
    if align:
        R, t, _ = umeyama(est, gt)
        est = est @ R.T + t
    return float(np.sqrt(((est - gt) ** 2).sum(1).mean()))


def _arc(gt: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))]


def first_distance_mask(gt: np.ndarray, first_m: float | None) -> np.ndarray:
    """True for samples whose GT path length from the first sample is <= first_m (None = all)."""
    return np.ones(len(gt), bool) if first_m is None else _arc(gt) <= first_m


def ate_first_m(est: np.ndarray, gt: np.ndarray, first_m: float | None = 100.0, align: bool = False) -> float:
    m = first_distance_mask(gt, first_m)
    return ate_rmse(est[m], gt[m], align=align)


def rpe_percent(est: np.ndarray, gt: np.ndarray, seg_m: float = 10.0) -> float:
    """Mean position-displacement error per GT path length over segments of >= seg_m, in percent."""
    s = _arc(gt)
    errs = []
    for i in range(len(gt)):
        j = int(np.searchsorted(s, s[i] + seg_m, side="left"))
        if j >= len(gt):
            break
        errs.append(np.linalg.norm((est[j] - est[i]) - (gt[j] - gt[i])) / (s[j] - s[i]))
    return float(np.mean(errs) * 100) if errs else 0.0
