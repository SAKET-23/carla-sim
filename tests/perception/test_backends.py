import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hybrid_av.perception.detection_torch import TorchDetector
from hybrid_av.perception.pillars import PillarConfig
from hybrid_av.perception.segmentation_torch import TorchSegmenter


class TinySeg(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.c = nn.Conv2d(3, k, 3, padding=1)

    def forward(self, x):
        return self.c(x)


class TinyHead(nn.Module):
    """Same I/O contract as the exported PointPillars network: 4x4 canvas → 2x2 output grid (stride 2), 2 anchors per cell."""

    def forward(self, feats, coords, num):
        n = 2 * 2 * 2
        return torch.full((n, 1), -10.0), torch.zeros((n, 7)), torch.zeros((n, 2))


CFG = PillarConfig(
    pc_range=(0, -0.32, -3, 0.64, 0.32, 1),
    voxel=(0.16, 0.16, 4.0),
    max_points=8,
    max_pillars=100,
    anchors=((3.9, 1.6, 1.56, -1.78),),
    rotations=(0.0, 1.5708),
    score_thresh=0.3,
    nms_iou=0.01,
    max_det=10,
)


def test_torch_segmenter_shape_and_dtype():
    torch.manual_seed(0)
    seg = TorchSegmenter(TinySeg(), mean=(0.5,) * 3, std=(0.25,) * 3, input_hw=(64, 128))
    out = seg(np.random.default_rng(0).integers(0, 255, (72, 128, 3), dtype=np.uint8))
    assert out.shape == (72, 128) and out.dtype == np.uint8 and out.max() < 5


def test_torch_detector_contract_with_no_detections():
    det = TorchDetector(TinyHead(), CFG)
    assert det(np.array([[0.3, 0.0, 0.0, 0.1]], np.float32)) == []


def test_torch_detector_empty_cloud_skips_network():
    class Boom(nn.Module):
        def forward(self, *a):
            raise AssertionError("network must not run")

    assert TorchDetector(Boom(), CFG)(np.zeros((0, 4), np.float32)) == []


def test_pointpillars_net_output_shapes():
    from hybrid_av.perception.pillars import PointPillarsNet, make_anchors, voxelize

    cfg = PillarConfig(
        pc_range=(-10.24, -10.24, -3, 10.24, 10.24, 3),
        voxel=(0.16, 0.16, 6.0),
        max_points=32,
        max_pillars=500,
        anchors=((3.9, 1.6, 1.56, -1.0), (0.8, 0.6, 1.73, -0.6), (1.76, 0.6, 1.73, -0.6)),
        rotations=(0.0, 1.5708),
        score_thresh=0.3,
        nms_iou=0.01,
        max_det=50,
    )
    pts = np.random.default_rng(0).uniform([-10, -10, -2, 0], [10, 10, 1, 1], (2000, 4)).astype(np.float32)
    f, c, n = voxelize(pts, cfg)
    cls, box, d = PointPillarsNet(cfg).eval()(torch.from_numpy(f), torch.from_numpy(c), torch.from_numpy(n))
    A = make_anchors(cfg, grid_hw=(64, 64)).shape[0]  # 128×128 canvas, backbone output stride 2 → 64×64
    assert cls.shape == (A, 3) and box.shape == (A, 7) and d.shape == (A, 2)


def test_openpcdet_state_dict_roundtrip(tmp_path):
    from hybrid_av.perception.pillars import PointPillarsNet, load_openpcdet_state_dict

    cfg = PillarConfig(pc_range=(-5.12, -5.12, -1, 5.12, 5.12, 5), voxel=(0.16, 0.16, 6.0), max_points=4,
                       max_pillars=10, anchors=((3.9, 1.6, 1.56, 0.78),) * 3, rotations=(0.0, 1.5708),
                       score_thresh=0.3, nms_iou=0.01, max_det=5)  # fmt: skip
    a, b = PointPillarsNet(cfg), PointPillarsNet(cfg)
    sd = {"vfe.pfn_layers.0.linear.weight": a.vfe.pfn_layers[0].linear.weight}
    sd = {**{k: v for k, v in a.state_dict().items()}, **sd}
    torch.save({"model_state": sd}, tmp_path / "ck.pth")
    load_openpcdet_state_dict(b, tmp_path / "ck.pth")
    assert torch.equal(a.dense_head.conv_cls.weight, b.dense_head.conv_cls.weight)


def test_default_pillar_config_matches_spec():
    cfg = PillarConfig()
    assert cfg.pc_range == (-51.2, -51.2, -1.0, 51.2, 51.2, 5.0) and cfg.voxel == (0.16, 0.16, 6.0)
    assert cfg.anchors[0] == (3.9, 1.6, 1.56, 0.78) and len(cfg.anchors) == 3


def test_yaml_pillars_equal_defaults():
    import yaml

    d = yaml.safe_load(Path("configs/perception/detection3d.yaml").read_text())["pillars"]
    assert PillarConfig.from_dict(d) == PillarConfig()


def test_perception_import_is_light():
    import subprocess

    code = "import sys, hybrid_av.perception, hybrid_av.perception.base; print(','.join(m for m in ('tensorrt','carla','rclpy') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_trt_modules_import_without_tensorrt():
    import hybrid_av.perception.detection_trt
    import hybrid_av.perception.segmentation_trt  # noqa

    assert "tensorrt" not in sys.modules


def test_trt_backend_missing_engine_message():
    from hybrid_av.perception.base import make_detector

    with pytest.raises(FileNotFoundError, match="export_models.py --build-engines"):
        make_detector({"backend": "trt", "engine": "/nonexistent/pp.engine"})


def test_trt_segmenter_missing_engine_message():
    from hybrid_av.perception.base import make_segmenter

    with pytest.raises(FileNotFoundError, match="export_models.py --build-engines"):
        make_segmenter({"backend": "trt", "engine": "/nonexistent/seg.engine"})


def test_factory_builds_torch_detector_from_yaml_pillars(tmp_path):
    from hybrid_av.perception.base import make_detector
    from hybrid_av.perception.pillars import PointPillarsNet

    cfg = PillarConfig(pc_range=(-5.12, -5.12, -1, 5.12, 5.12, 5), voxel=(0.16, 0.16, 6.0), max_points=4,
                       max_pillars=10, anchors=((3.9, 1.6, 1.56, 0.78),) * 3, rotations=(0.0, 1.5708),
                       score_thresh=0.3, nms_iou=0.01, max_det=5)  # fmt: skip
    torch.save({"model_state": PointPillarsNet(cfg).state_dict()}, tmp_path / "c.pth")
    det = make_detector({"backend": "torch", "checkpoint": str(tmp_path / "c.pth"),
                         "pillars": {"pc_range": [-5.12, -5.12, -1, 5.12, 5.12, 5], "voxel": [0.16, 0.16, 6.0],
                                     "max_points": 4, "max_pillars": 10,
                                     "anchors": [[3.9, 1.6, 1.56, 0.78]] * 3, "rotations": [0.0, 1.5708],
                                     "score_thresh": 0.3, "nms_iou": 0.01, "max_det": 5}})  # fmt: skip
    assert det(np.zeros((0, 4), np.float32)) == []


@pytest.mark.gpu
def test_trt_detector_runs_on_gpu(built_engine_path):
    from hybrid_av.perception.detection_trt import TrtDetector

    assert isinstance(TrtDetector(built_engine_path, CFG)(np.zeros((0, 4), np.float32)), list)
