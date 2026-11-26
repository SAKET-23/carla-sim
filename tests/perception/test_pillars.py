import numpy as np
import pytest

from hybrid_av.perception.pillars import (
    PillarConfig,
    decode_boxes,
    encode_boxes,
    make_anchors,
    postprocess,
    voxelize,
)

CFG = PillarConfig(
    pc_range=(0, -40, -3, 70.4, 40, 1),
    voxel=(0.16, 0.16, 4.0),
    max_points=32,
    max_pillars=16000,
    anchors=((3.9, 1.6, 1.56, -1.78),),
    rotations=(0.0, np.pi / 2),
    score_thresh=0.3,
    nms_iou=0.01,
    max_det=100,
)


def test_voxelize_single_pillar_features():
    pts = np.array([[1.0, 1.0, 0.0, 0.5], [1.1, 1.05, 0.2, 0.7]], np.float32)
    f, c, n = voxelize(pts, CFG)
    assert f.shape == (1, 32, 10) and n.tolist() == [2]
    ix, iy = int(1.0 / 0.16), int((1.0 + 40) / 0.16)
    assert c.tolist() == [[iy, ix]]
    np.testing.assert_allclose(f[0, 0, 4:7], [1.0 - 1.05, 1.0 - 1.025, 0.0 - 0.1], atol=1e-6)
    np.testing.assert_allclose(f[0, 0, 7], 1.0 - (ix + 0.5) * 0.16, atol=1e-6)
    assert np.all(f[0, 2:] == 0)


def test_voxelize_drops_out_of_range_and_caps():
    pts = np.array([[-1, 0, 0, 0], [80, 0, 0, 0], [5, 0, 5, 0]], np.float32)
    assert voxelize(pts, CFG)[0].shape[0] == 0
    many = np.tile(np.array([[1.0, 1.0, 0.0, 0.0]], np.float32), (100, 1))
    assert voxelize(many, CFG)[2].tolist() == [32]


def test_voxelize_empty_cloud():
    f, c, n = voxelize(np.zeros((0, 4), np.float32), CFG)
    assert f.shape == (0, 32, 10) and c.shape == (0, 2) and n.shape == (0,)


def test_encode_decode_roundtrip():
    anchors = make_anchors(CFG, grid_hw=(4, 4))
    assert anchors.shape == (4 * 4 * 2, 7)
    rng = np.random.default_rng(0)
    boxes = anchors + rng.normal(scale=[0.3, 0.3, 0.1, 0.2, 0.1, 0.1, 0.2], size=anchors.shape)
    np.testing.assert_allclose(decode_boxes(encode_boxes(boxes, anchors), anchors), boxes, atol=1e-5)


def test_postprocess_nms_keeps_best_of_overlapping():
    anchors = np.array(
        [
            [10, 0, -1, 3.9, 1.6, 1.56, 0.0],
            [10.2, 0, -1, 3.9, 1.6, 1.56, 0.0],
            [30, 5, -1, 3.9, 1.6, 1.56, 0.0],
        ]
    )
    cls = np.array([[3.0], [1.0], [-5.0]])  # sigmoid 0.95, 0.73, 0.007
    out = postprocess(cls, np.zeros((3, 7)), np.array([[1.0, 0.0]] * 3), anchors, CFG)
    assert len(out) == 1 and out[0].x == pytest.approx(10.0)
    assert out[0].score == pytest.approx(1 / (1 + np.exp(-3)))
