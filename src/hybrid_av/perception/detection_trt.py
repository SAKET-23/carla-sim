"""TensorRT PointPillars backend: numpy voxelisation and decoding around an engine (detection-3d.md §5.2)."""

from pathlib import Path

import numpy as np
import torch

from hybrid_av.perception.base import Box3D
from hybrid_av.perception.pillars import PillarConfig, detect
from hybrid_av.perception.trt_common import Engine, require_engine


class TrtDetector:
    """Engine I/O names follow export: inputs (pillar_features, coords, num_points); outputs in head order."""

    def __init__(self, engine_path: Path, pillar_cfg: PillarConfig) -> None:
        self.engine = Engine(require_engine(engine_path))  # tensorrt imported here, after the existence check
        self.cfg = pillar_cfg

    def _net(self, feats, coords, num):
        out = self.engine(
            pillar_features=torch.from_numpy(feats).cuda(),
            coords=torch.from_numpy(coords).cuda(),
            num_points=torch.from_numpy(num).cuda(),
        )
        return tuple(out[n].float().cpu().numpy() for n in ("cls_logits", "box_deltas", "dir_logits"))

    def __call__(self, points: np.ndarray) -> list[Box3D]:
        return detect(points, self.cfg, self._net)


def build(cfg: dict) -> TrtDetector:
    path = require_engine(Path(cfg["engine"]))  # first: no other key is read for a missing engine
    return TrtDetector(path, PillarConfig.from_dict(cfg.get("pillars", {})))
