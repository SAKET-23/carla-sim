"""Perception types, protocols and factories (interfaces.md §13)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hybrid_av.geometry import rotate, world_to_base
from hybrid_av.perception.classes import CARLA_TAG_TO_TRAIN, TRAIN_CLASSES
from hybrid_av.types import EgoState, ObjectClass, ObjectState


@dataclass(frozen=True, slots=True)
class Box3D:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float  # base_link frame
    cls: ObjectClass
    score: float


@dataclass(frozen=True, slots=True)
class RadarDetection:
    depth: float
    azimuth: float
    altitude: float
    velocity: float  # sensor frame, CARLA semantics: relative radial, negative toward the sensor
    sensor_id: str


class Segmenter(Protocol):
    classes: tuple[str, ...]

    def __call__(self, rgb: np.ndarray) -> np.ndarray: ...  # (H,W,3) uint8 RGB -> (H,W) uint8 class ids


class Detector3D(Protocol):
    def __call__(self, points: np.ndarray) -> list[Box3D]: ...  # (N,4) float32 x,y,z,intensity in base_link


class GtSegmenter:
    """CARLA semantic camera passthrough: (H,W) tag ids or (H,W,3) RGB with the tag in channel 0."""

    classes = TRAIN_CLASSES

    def __call__(self, img: np.ndarray) -> np.ndarray:
        tags = img[..., 0] if img.ndim == 3 else img
        return CARLA_TAG_TO_TRAIN[tags]


class GtDetector:
    """Ground-truth objects in base_link with the detection-3d.md label rule (>= min_points points, <= 70 m)."""

    def __init__(self, min_points: int = 10) -> None:
        self.min_points = min_points
        self._boxes: list[Box3D] = []

    def set_ground_truth(self, objects: Sequence[ObjectState], ego: EgoState) -> None:
        self._boxes = []
        for o in objects:
            x, y = world_to_base(np.array([o.x, o.y]), ego)
            if np.hypot(x, y) > 70.0:
                continue
            self._boxes.append(
                Box3D(float(x), float(y), o.z, o.length, o.width, o.height, o.yaw - ego.psi, o.cls, 1.0)
            )

    def __call__(self, points: np.ndarray) -> list[Box3D]:
        out = []
        for b in self._boxes:
            d = rotate(points[:, :2] - np.array([b.x, b.y]), -b.yaw)  # box frame
            inside = (
                (np.abs(d[:, 0]) <= b.length / 2)
                & (np.abs(d[:, 1]) <= b.width / 2)
                & (np.abs(points[:, 2] - b.z) <= b.height / 2)
            )
            if inside.sum() >= self.min_points:
                out.append(b)
        return out


def make_segmenter(cfg: dict) -> Segmenter:
    backend = cfg["backend"]
    if backend == "gt":
        return GtSegmenter()
    if backend == "torch":
        from hybrid_av.perception import segmentation_torch

        return segmentation_torch.build(cfg)
    if backend == "trt":
        from hybrid_av.perception import segmentation_trt

        return segmentation_trt.build(cfg)
    raise ValueError(f"unknown segmenter backend {backend!r}")


def make_detector(cfg: dict) -> Detector3D:
    backend = cfg["backend"]
    if backend == "gt":
        return GtDetector(cfg.get("min_points", 10))
    if backend == "torch":
        from hybrid_av.perception import detection_torch

        return detection_torch.build(cfg)
    if backend == "trt":
        from hybrid_av.perception import detection_trt

        return detection_trt.build(cfg)
    raise ValueError(f"unknown detector backend {backend!r}")
