import numpy as np
import pytest

from hybrid_av.eval.ate import associate, ate_first_m, ate_rmse, rpe_percent, umeyama


def path(n=200):
    t = np.linspace(0, 20, n)
    return np.c_[10 * np.cos(t / 5), 10 * np.sin(t / 5), 0.1 * t]


def test_rigid_transform_gives_zero_ate():
    gt = path()
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    est = gt @ R.T + [5.0, -3.0, 1.0]
    assert ate_rmse(est, gt) < 1e-9
    assert ate_rmse(est, gt, align=False) > 1.0


def test_known_noise_level():
    gt = path(2000)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=gt.shape)
    noise -= noise.mean(0)
    noise *= 0.1 / np.sqrt((noise**2).sum(1).mean())
    assert ate_rmse(gt + noise, gt) == pytest.approx(0.1, rel=0.05)


def test_umeyama_recovers_rotation():
    gt = path()
    R, t, s = umeyama(gt, gt + [1, 2, 3])
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(t, [1, 2, 3], atol=1e-9)
    assert s == 1.0


def test_umeyama_with_scale_2d():
    gt = path()[:, :2]
    R, _t, s = umeyama(gt, 2.0 * gt, with_scale=True)
    assert s == pytest.approx(2.0) and R.shape == (2, 2)


def test_ate_first_m_scores_first_100_m_only():
    gt = np.c_[np.linspace(0, 200, 401), np.zeros(401), np.zeros(401)]
    est = gt.copy()
    est[gt[:, 0] > 100, 1] += 1.0  # error appears only after 100 m
    assert ate_first_m(est, gt) < 1e-9 and ate_first_m(est, gt, first_m=None) > 0.1
    assert ate_first_m(est, gt, align=True) < 1e-9


def test_rpe_one_percent_scale_drift():
    gt = np.c_[np.linspace(0, 200, 401), np.zeros(401), np.zeros(401)]
    assert rpe_percent(gt * 1.01, gt) == pytest.approx(1.0, rel=1e-6)


def test_associate_nearest_timestamp():
    i, j = associate(np.array([0.0, 0.05, 0.1]), np.array([0.001, 0.049, 0.2]))
    assert i.tolist() == [0, 1] and j.tolist() == [0, 1]
