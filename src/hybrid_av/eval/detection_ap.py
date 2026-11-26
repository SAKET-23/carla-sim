"""KITTI-style AP_3D with 40-point interpolation (R40); metrics.md P2. Not bit-identical to the official devkit."""

import numpy as np

from hybrid_av.geometry import iou_3d
from hybrid_av.perception.base import Box3D
from hybrid_av.types import ObjectClass


def average_precision_r40(
    gts: list[list[Box3D]],
    dets: list[list[Box3D]],
    cls: ObjectClass,
    iou_thresh: float = 0.5,
    max_range: float = 50.0,
) -> float:
    """Per-frame lists; boxes beyond `max_range` (BEV distance from the origin) are dropped on both sides."""

    def keep(boxes):
        return [b for b in boxes if b.cls == cls and np.hypot(b.x, b.y) <= max_range]

    n_gt, scored = 0, []  # scored: (score, is_tp) over all frames
    for g, d in zip(gts, dets):
        g, d = keep(g), sorted(keep(d), key=lambda b: -b.score)
        n_gt += len(g)
        used = [False] * len(g)
        for det in d:
            ious = [(iou_3d(det, gt) if not used[k] else -1.0) for k, gt in enumerate(g)]
            k = int(np.argmax(ious)) if ious else -1
            tp = k >= 0 and ious[k] >= iou_thresh
            if tp:
                used[k] = True
            scored.append((det.score, tp))
    if n_gt == 0 or not scored:
        return 0.0
    tp = np.array([t for _, t in sorted(scored, key=lambda s: -s[0])], float)
    ctp = np.cumsum(tp)
    precision = ctp / np.arange(1, len(tp) + 1)
    recall = ctp / n_gt
    interp = np.maximum.accumulate(precision[::-1])[::-1]  # max precision at recall' >= r
    ap = 0.0
    for r in np.arange(1, 41) / 40:
        i = np.searchsorted(recall, r - 1e-12, side="left")  # first operating point reaching r
        ap += interp[i] if i < len(recall) else 0.0
    return float(ap / 40)
