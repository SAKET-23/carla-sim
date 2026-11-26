import math

import numpy as np
import pytest
from hybrid_av_bringup.tf_math import quat_from_rpy


def _matrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rpy_matrix(r, p, y):
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


@pytest.mark.parametrize(
    "rpy", [(0, 0, 0), (0, 0, math.pi), (-math.pi / 2, 0, -math.pi / 2), (0.1, -0.2, 0.3)]
)
def test_quat_matches_rotation_matrix(rpy):
    q = quat_from_rpy(*rpy)
    assert sum(v * v for v in q) == pytest.approx(1.0)
    assert np.allclose(_matrix(q), _rpy_matrix(*rpy), atol=1e-12)
