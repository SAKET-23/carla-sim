"""TensorRT segmentation backend; same pre/post-processing as the torch path (segmentation.md §5)."""

from pathlib import Path

import numpy as np

from hybrid_av.perception.classes import TRAIN_CLASSES
from hybrid_av.perception.segmentation_torch import postprocess, preprocess
from hybrid_av.perception.trt_common import Engine, require_engine


class TrtSegmenter:
    """Engine I/O names follow export: input `image`, output `logits`."""

    classes = TRAIN_CLASSES

    def __init__(self, engine_path: Path, mean, std, input_hw) -> None:
        self.engine = Engine(require_engine(engine_path))
        self.mean, self.std, self.input_hw = mean, std, input_hw

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = preprocess(rgb, self.mean, self.std, self.input_hw, "cuda")
        return postprocess(self.engine(image=x)["logits"].float(), rgb.shape[:2])


def build(cfg: dict) -> TrtSegmenter:
    path = require_engine(Path(cfg["engine"]))
    return TrtSegmenter(path, cfg["mean"], cfg["std"], cfg["input_hw"])
