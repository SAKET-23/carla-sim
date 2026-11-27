"""Segmentation pair writer: RGB PNG plus train-id PNG (255 = ignore)."""

from pathlib import Path

import numpy as np
from PIL import Image

from hybrid_av.perception.classes import CARLA_TAG_TO_TRAIN


class SegWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for d in ("images", "labels"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def write_pair(self, idx: int, rgb: np.ndarray, tags: np.ndarray) -> None:
        Image.fromarray(np.asarray(rgb, np.uint8)).save(self.root / f"images/{idx:06d}.png")
        Image.fromarray(CARLA_TAG_TO_TRAIN[np.asarray(tags, np.uint8)]).save(
            self.root / f"labels/{idx:06d}.png"
        )
