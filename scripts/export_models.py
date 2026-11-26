"""Export the perception networks to ONNX and build TensorRT engines (paths from configs/perception/*.yaml)."""

import argparse
from pathlib import Path

import torch
import yaml

CFG = Path(__file__).resolve().parents[1] / "configs" / "perception"


def _onnx_path(cfg: dict) -> Path:
    return Path(cfg["engine"]).with_suffix(".onnx")


def export_detector(cfg: dict) -> Path:
    from hybrid_av.perception.export_onnx import export_onnx
    from hybrid_av.perception.pillars import PillarConfig, PointPillarsNet, load_openpcdet_state_dict

    pc = PillarConfig.from_dict(cfg.get("pillars", {}))
    net = PointPillarsNet(pc).eval()
    load_openpcdet_state_dict(net, Path(cfg["checkpoint"]))
    p = 100  # example pillar count; the P axis is dynamic
    inputs = (
        torch.zeros(p, pc.max_points, 10),
        torch.stack([torch.arange(p) // 640, torch.arange(p) % 640], 1).int(),
        torch.ones(p, dtype=torch.int32),
    )
    names = ["pillar_features", "coords", "num_points"]
    return export_onnx(
        net, inputs, _onnx_path(cfg), names, ["cls_logits", "box_deltas", "dir_logits"],
        dynamic_axes={n: {0: "P"} for n in names},
    )  # fmt: skip


def export_segmenter(cfg: dict) -> Path:
    from hybrid_av.perception.export_onnx import export_onnx
    from hybrid_av.perception.segmentation_torch import load_segmentation_model

    h, w = cfg["input_hw"]
    return export_onnx(
        load_segmentation_model(cfg), (torch.zeros(1, 3, h, w),), _onnx_path(cfg), ["image"], ["logits"]
    )


def build_engines(det: dict, seg: dict) -> None:
    from hybrid_av.perception.build_engine import build_engine
    from hybrid_av.perception.pillars import PillarConfig

    pc = PillarConfig.from_dict(det.get("pillars", {}))
    f = pc.max_points
    profiles = {  # P in [1, max_pillars]
        "pillar_features": ((1, f, 10), (6000, f, 10), (pc.max_pillars, f, 10)),
        "coords": ((1, 2), (6000, 2), (pc.max_pillars, 2)),
        "num_points": ((1,), (6000,), (pc.max_pillars,)),
    }
    build_engine(_onnx_path(det), Path(det["engine"]), profiles=profiles)
    build_engine(_onnx_path(seg), Path(seg["engine"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--export-onnx", action="store_true", help="checkpoints -> ONNX (next to the engine paths)"
    )
    ap.add_argument("--build-engines", action="store_true", help="ONNX -> TensorRT engines (pod only)")
    args = ap.parse_args()
    det = yaml.safe_load((CFG / "detection3d.yaml").read_text())
    seg = yaml.safe_load((CFG / "segmentation.yaml").read_text())
    if args.export_onnx:
        print(export_detector(det), export_segmenter(seg))
    if args.build_engines:
        build_engines(det, seg)
    if not (args.export_onnx or args.build_engines):
        ap.error("give --export-onnx and/or --build-engines")


if __name__ == "__main__":
    main()
