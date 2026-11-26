"""CARLA semantic tags, training classes and actor-to-ObjectClass mapping (segmentation.md §2)."""

import numpy as np

from hybrid_av.types import ObjectClass

# VERIFY-IN-G2: tag ids, truck/van/bus and bike/motorcycle blueprint lists are from CARLA 0.9.16 documentation,
# not checked against a running server (integration-check.md, semantic-tags item).
CARLA_TAGS: dict[int, str] = {
    0: "Unlabeled", 1: "Roads", 2: "SideWalks", 3: "Building", 4: "Wall", 5: "Fence", 6: "Pole",
    7: "TrafficLight", 8: "TrafficSign", 9: "Vegetation", 10: "Terrain", 11: "Sky", 12: "Pedestrian",
    13: "Rider", 14: "Car", 15: "Truck", 16: "Bus", 17: "Train", 18: "Motorcycle", 19: "Bicycle",
    20: "Static", 21: "Dynamic", 22: "Other", 23: "Water", 24: "RoadLine", 25: "Ground", 26: "Bridge",
    27: "RailTrack", 28: "GuardRail",
}  # fmt: skip

TRAIN_CLASSES = (
    "road",
    "sidewalk",
    "vehicle",
    "person",
    "static_obstacle",
    "vegetation_terrain",
    "sky",
    "other",
)
IGNORE_ID = 255

_TRAIN_TAGS = {
    0: (1, 24),
    1: (2, 25),
    2: (14, 15, 16, 17, 18, 19),
    3: (12, 13),
    4: (3, 4, 5, 6, 7, 8, 20, 26, 28),
    5: (9, 10),
    6: (11,),
    7: (21, 22, 23, 27),
}
CARLA_TAG_TO_TRAIN = np.full(256, IGNORE_ID, np.uint8)
for _tid, _tags in _TRAIN_TAGS.items():
    CARLA_TAG_TO_TRAIN[list(_tags)] = _tid

_TRUCKS = (
    "carlacola",
    "cybertruck",
    "firetruck",
    "ambulance",
    "sprinter",
    "volkswagen.t2",
    "fusorosa",
    "european_hgv",
)
_BIKES = ("crossbike", "century", "omafiets", "low_rider", "ninja", "vespa", "yzf")


def CARLA_ACTOR_TO_CLASS(type_id: str) -> ObjectClass:
    if type_id.startswith("walker."):
        return ObjectClass.PEDESTRIAN
    if any(k in type_id for k in _BIKES):
        return ObjectClass.CYCLIST
    if any(k in type_id for k in _TRUCKS):
        return ObjectClass.TRUCK
    return ObjectClass.CAR


# segmentation.md §7: bottom 40 % trapezoid, (x, y) fractions of width and height; checked in G2 (IC-15).
DRIVABLE_ROI = ((0.20, 1.00), (0.80, 1.00), (0.56, 0.60), (0.44, 0.60))


def drivable_ratio(train_ids: np.ndarray) -> float:
    h, w = train_ids.shape
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    px, py = np.meshgrid(u, v)
    signs = []
    for (x0, y0), (x1, y1) in zip(DRIVABLE_ROI, DRIVABLE_ROI[1:] + DRIVABLE_ROI[:1]):
        signs.append((x1 - x0) * (py - y0) - (y1 - y0) * (px - x0))
    inside = np.all([s >= 0 for s in signs], axis=0) | np.all([s <= 0 for s in signs], axis=0)
    n = int(inside.sum())
    return float((train_ids[inside] == 0).sum() / n) if n else 0.0
