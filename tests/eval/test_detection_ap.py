import pytest

from hybrid_av.eval.detection_ap import average_precision_r40
from hybrid_av.perception.base import Box3D
from hybrid_av.types import ObjectClass


def car(x, y=0.0, s=1.0):
    return Box3D(x=x, y=y, z=0, length=4, width=2, height=1.5, yaw=0, cls=ObjectClass.CAR, score=s)


def test_hand_computed_ap():
    # GT A at x=10, B at x=30. Dets: 0.9 on A, 0.8 false positive, 0.7 on B.
    # PR points: (P=1, R=.5), (P=.5, R=.5), (P=2/3, R=1). Interpolated: 20 points at 1.0, 20 at 2/3 → 0.8333
    ap = average_precision_r40(
        [[car(10), car(30)]], [[car(10, s=0.9), car(20, y=10, s=0.8), car(30, s=0.7)]], ObjectClass.CAR
    )
    assert ap == pytest.approx(0.8333, abs=1e-3)


def test_range_filter_and_double_match():
    assert average_precision_r40([[car(60)]], [[car(60)]], ObjectClass.CAR) == 0.0  # nothing inside 50 m
    # duplicate detection is a false positive after recall already reached 1 → interpolated precision stays 1.0
    assert average_precision_r40(
        [[car(10)]], [[car(10, s=0.9), car(10, s=0.8)]], ObjectClass.CAR
    ) == pytest.approx(1.0)


def test_perfect_and_empty():
    assert average_precision_r40(
        [[car(10)], [car(20)]], [[car(10)], [car(20)]], ObjectClass.CAR
    ) == pytest.approx(1.0)
    assert average_precision_r40([[car(10)]], [[]], ObjectClass.CAR) == 0.0


def test_other_class_ignored():
    ped = Box3D(x=10, y=0, z=0, length=4, width=2, height=1.5, yaw=0, cls=ObjectClass.PEDESTRIAN, score=0.9)
    assert average_precision_r40([[car(10)]], [[ped]], ObjectClass.CAR) == 0.0
