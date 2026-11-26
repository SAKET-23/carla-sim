import sys

import numpy as np

from hybrid_av.perception.base import make_detector, make_segmenter
from hybrid_av.types import EgoState, ObjectClass, ObjectState


def test_package_import_is_light():
    import hybrid_av.perception  # noqa: F401

    assert "tensorrt" not in sys.modules and "carla" not in sys.modules


def test_gt_segmenter_accepts_both_layouts():
    seg = make_segmenter({"backend": "gt"})
    tags = np.array([[1, 24], [14, 11]], np.uint8)
    out2 = seg(tags)
    out3 = seg(np.stack([tags, np.zeros_like(tags), np.zeros_like(tags)], -1))
    assert out2.shape == (2, 2) and out2.dtype == np.uint8 and np.array_equal(out2, out3)


def test_gt_detector_filters_by_points_and_converts_frame():
    det = make_detector({"backend": "gt"})  # min_points 10
    ego = EgoState(t=0, x=100, y=0, psi=0, vx=0, vy=0, r=0)
    kw = {
        "cls": ObjectClass.CAR,
        "z": 0.8,
        "yaw": 0,
        "length": 4,
        "width": 2,
        "height": 1.5,
        "vx": 0,
        "vy": 0,
    }
    visible = ObjectState(id=1, x=110, y=0, **kw)
    hidden = ObjectState(id=2, x=130, y=0, **kw)
    det.set_ground_truth([visible, hidden], ego)
    pts = np.zeros((10, 4), np.float32)
    pts[:, 0] = 10.0
    pts[:, 2] = 0.8  # 10 points inside the visible box
    boxes = det(pts)
    assert len(boxes) == 1 and abs(boxes[0].x - 10.0) < 1e-9 and boxes[0].score == 1.0
