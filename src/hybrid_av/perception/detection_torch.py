"""PyTorch PointPillars backend; pre/post-processing is shared with the TensorRT backend (pillars.py)."""

import numpy as np
import torch
from torch import nn

from hybrid_av.perception.base import Box3D
from hybrid_av.perception.pillars import PillarConfig, detect


class TorchDetector:
    """`model` has the exported ONNX contract: (features, coords, num_points) -> (cls_logits, box_deltas, dir_logits)."""

    def __init__(self, model: nn.Module, pillar_cfg: PillarConfig, device: str = "cpu") -> None:
        self.model = model.eval().to(device)
        self.cfg = pillar_cfg
        self.device = device

    def _net(self, feats, coords, num):
        with torch.no_grad():
            out = self.model(
                torch.from_numpy(feats).to(self.device),
                torch.from_numpy(coords).to(self.device),
                torch.from_numpy(num).to(self.device),
            )
        return tuple(o.float().cpu().numpy() for o in out)

    def __call__(self, points: np.ndarray) -> list[Box3D]:
        return detect(points, self.cfg, self._net)


def build(cfg: dict) -> TorchDetector:
    """Factory entry: cfg keys `checkpoint` (OpenPCDet .pth), `pillars` (PillarConfig mapping), `device`."""
    from pathlib import Path

    from hybrid_av.perception.pillars import PointPillarsNet, load_openpcdet_state_dict

    pcfg = PillarConfig.from_dict(cfg.get("pillars", {}))
    net = PointPillarsNet(pcfg)
    load_openpcdet_state_dict(net, Path(cfg["checkpoint"]))
    return TorchDetector(net, pcfg, cfg.get("device", "cpu"))
