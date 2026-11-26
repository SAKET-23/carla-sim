import numpy as np

from hybrid_av.tracking.association import assign, mahalanobis_cost


def test_global_optimum_not_greedy():
    cost = np.array([[1.0, 2.0], [1.5, 8.0]])
    m, ut, ud = assign(cost, gate=9.21)
    assert sorted(m) == [(0, 1), (1, 0)] and ut == [] and ud == []


def test_gate_blocks_far_pairs():
    m, ut, ud = assign(np.array([[1.0, 50.0], [50.0, 20.0]]), gate=9.21)
    assert m == [(0, 0)] and ut == [1] and ud == [1]


def test_empty_inputs():
    assert assign(np.zeros((0, 3))) == ([], [], [0, 1, 2])
    assert assign(np.zeros((2, 0))) == ([], [0, 1], [])


def test_mahalanobis_identity_cov():
    c = mahalanobis_cost(
        np.array([[0.0, 0.0]]), np.eye(2)[None] * 0.5, np.array([[1.0, 1.0]]), np.eye(2)[None] * 0.5
    )
    assert c.shape == (1, 1) and np.isclose(c[0, 0], 2.0)
