"""OpenPCDet "custom dataset" writer: LiDAR-only, boxes in base_link (detection-3d.md §2)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hybrid_av.perception.base import Box3D
from hybrid_av.types import ObjectClass

_NAME = {
    ObjectClass.CAR: "Car",
    ObjectClass.TRUCK: "Car",  # trucks, buses and vans train as Car
    ObjectClass.PEDESTRIAN: "Pedestrian",
    ObjectClass.CYCLIST: "Cyclist",
}
_CLASS = {"Car": ObjectClass.CAR, "Pedestrian": ObjectClass.PEDESTRIAN, "Cyclist": ObjectClass.CYCLIST}


class DetectionWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for d in ("points", "labels", "ImageSets"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def write_frame(self, idx: int, points: np.ndarray, boxes: Sequence[Box3D]) -> None:
        np.save(self.root / f"points/{idx:06d}.npy", np.asarray(points, np.float32))
        lines = [
            f"{b.x:.6f} {b.y:.6f} {b.z:.6f} {b.length:.6f} {b.width:.6f} {b.height:.6f} {b.yaw:.6f} {_NAME[b.cls]}\n"
            for b in boxes
        ]  # KeyError for ObjectClass.UNKNOWN: unlabelled classes must not reach the dataset
        (self.root / f"labels/{idx:06d}.txt").write_text("".join(lines))

    def write_split(self, name: str, indices: Sequence[int]) -> None:
        (self.root / f"ImageSets/{name}.txt").write_text("".join(f"{i:06d}\n" for i in indices))


def read_labels(path: Path) -> list[Box3D]:
    out = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if f:
            x, y, z, dx, dy, dz, yaw = map(float, f[:7])
            out.append(Box3D(x, y, z, dx, dy, dz, yaw, _CLASS[f[7]], 1.0))
    return out
