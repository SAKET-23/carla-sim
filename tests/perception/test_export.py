import sys

import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx")
from hybrid_av.perception.export_onnx import export_onnx


def test_export_tiny_seg_model(tmp_path):
    m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 5, 1)).eval()
    p = export_onnx(m, (torch.zeros(1, 3, 64, 128),), tmp_path / "seg.onnx", ["image"], ["logits"])
    g = onnx.load(p).graph
    assert [i.name for i in g.input] == ["image"] and [o.name for o in g.output] == ["logits"]


def test_export_dynamic_axis(tmp_path):
    class M(nn.Module):
        def forward(self, x):
            return x.sum(dim=1)

    p = export_onnx(
        M(),
        (torch.zeros(10, 8, 10),),
        tmp_path / "pp.onnx",
        ["pillars"],
        ["out"],
        dynamic_axes={"pillars": {0: "P"}, "out": {0: "P"}},
    )
    dim0 = onnx.load(p).graph.input[0].type.tensor_type.shape.dim[0]
    assert dim0.dim_param == "P"


def test_export_pointpillars_net_has_scatter_and_dynamic_pillars(tmp_path):
    import numpy as np

    from hybrid_av.perception.pillars import PillarConfig, PointPillarsNet, voxelize

    cfg = PillarConfig(pc_range=(-5.12, -5.12, -1, 5.12, 5.12, 5), voxel=(0.16, 0.16, 6.0), max_points=8,
                       max_pillars=100, anchors=((3.9, 1.6, 1.56, 0.78),) * 3)  # fmt: skip
    pts = np.random.default_rng(0).uniform([-5, -5, 0, 0], [5, 5, 2, 1], (500, 4)).astype(np.float32)
    f, c, n = (torch.from_numpy(a) for a in voxelize(pts, cfg))
    p = export_onnx(
        PointPillarsNet(cfg).eval(),
        (f, c, n),
        tmp_path / "pp.onnx",
        ["pillar_features", "coords", "num_points"],
        ["cls_logits", "box_deltas", "dir_logits"],
        dynamic_axes={k: {0: "P"} for k in ("pillar_features", "coords", "num_points")},
    )
    assert "ScatterND" in {nd.op_type for nd in onnx.load(p).graph.node}


def test_build_engine_module_is_lazy():
    import hybrid_av.perception.build_engine  # noqa: F401

    assert "tensorrt" not in sys.modules


@pytest.mark.gpu
def test_build_engine_on_gpu(tmp_path):
    from hybrid_av.perception.build_engine import build_engine

    m = nn.Sequential(nn.Conv2d(3, 5, 1)).eval()
    o = export_onnx(m, (torch.zeros(1, 3, 32, 32),), tmp_path / "t.onnx", ["image"], ["logits"])
    e = build_engine(o, tmp_path / "t.engine", fp16=True)
    assert e.exists() and (tmp_path / "t.engine.json").exists()
