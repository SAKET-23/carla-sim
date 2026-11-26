import numpy as np

from hybrid_av.policy.network import scale_action, unscale_action
from hybrid_av.types import PolicyAction


def test_endpoints_and_midpoint():
    assert scale_action(np.array([-1.0, -1, -1])) == PolicyAction(0.0, -3.5, 0.0)
    assert scale_action(np.array([1.0, 1, 1])) == PolicyAction(1.2, 3.5, 1.0)
    a = scale_action(np.zeros(3))
    assert np.allclose([a.alpha_v, a.delta_y, a.brake], [0.6, 0.0, 0.5])


def test_roundtrip():
    y = np.array([0.3, -0.7, 0.1])
    np.testing.assert_allclose(unscale_action(scale_action(y)), y, atol=1e-6)


def test_unscale_clips_boundary():
    y = unscale_action(PolicyAction(1.2, -3.5, 1.0))
    assert np.all(np.abs(y) < 1.0) and np.all(np.abs(y) >= 1 - 1e-6 - 1e-12)
