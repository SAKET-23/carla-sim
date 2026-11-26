"""PointPillars pre- and post-processing in numpy (detection-3d.md §3, §5; OpenPCDet conventions)."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hybrid_av.geometry import iou_bev, wrap_angle
from hybrid_av.perception.base import Box3D
from hybrid_av.types import ObjectClass

_CLASSES = (ObjectClass.CAR, ObjectClass.PEDESTRIAN, ObjectClass.CYCLIST)
_DIR_OFFSET = 0.785  # OpenPCDet dir_offset; dir_limit_offset is 0
_NMS_PRE = 4096  # OpenPCDet nms_pre_maxsize


@dataclass(frozen=True)
class PillarConfig:
    """Defaults are detection-3d.md §3 (360° range, 640 x 640 canvas, anchor z centres = h/2 above ground)."""

    pc_range: tuple[float, float, float, float, float, float] = (-51.2, -51.2, -1.0, 51.2, 51.2, 5.0)
    voxel: tuple[float, float, float] = (0.16, 0.16, 6.0)
    max_points: int = 32
    max_pillars: int = 16000
    anchors: tuple[tuple[float, float, float, float], ...] = (  # (l, w, h, z_centre) per class
        (3.9, 1.6, 1.56, 0.78),
        (0.8, 0.6, 1.73, 0.87),
        (1.76, 0.6, 1.73, 0.87),
    )
    rotations: tuple[float, ...] = (0.0, np.pi / 2)
    score_thresh: float = 0.3
    nms_iou: float = 0.01
    max_det: int = 100

    @classmethod
    def from_dict(cls, d: dict) -> "PillarConfig":
        """YAML mapping (lists) to a config; missing keys keep the defaults."""

        def tup(v):
            return tuple(tup(x) for x in v) if isinstance(v, list) else v

        return cls(**{k: tup(v) for k, v in d.items()})

    @property
    def canvas_hw(self) -> tuple[int, int]:
        x0, y0, _, x1, y1, _ = self.pc_range
        return round((y1 - y0) / self.voxel[1]), round((x1 - x0) / self.voxel[0])


def voxelize(points: np.ndarray, cfg: PillarConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(features (P, max_points, 10) float32, coords (P, 2) int32 [iy, ix], num_points (P,) int32)."""
    x0, y0, z0, x1, y1, z1 = cfg.pc_range
    vx, vy, vz = cfg.voxel
    nx = round((x1 - x0) / vx)
    pts = np.asarray(points, np.float64)
    keep = (
        (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        & (pts[:, 2] >= z0) & (pts[:, 2] < z1)
    )  # fmt: skip
    pts = pts[keep]
    ix = np.floor((pts[:, 0] - x0) / vx).astype(np.int64)
    iy = np.floor((pts[:, 1] - y0) / vy).astype(np.int64)
    order = np.argsort(iy * nx + ix, kind="stable")
    pts, ix, iy = pts[order], ix[order], iy[order]
    uniq, start, inv = np.unique(iy * nx + ix, return_index=True, return_inverse=True)
    rank = np.arange(len(pts)) - start[inv]
    sel = (rank < cfg.max_points) & (inv < cfg.max_pillars)
    pts, ix, iy, inv, rank = pts[sel], ix[sel], iy[sel], inv[sel], rank[sel]
    p = min(len(uniq), cfg.max_pillars)
    feats = np.zeros((p, cfg.max_points, 10), np.float32)
    counts = np.bincount(inv, minlength=p)
    mean = np.zeros((p, 3))
    np.add.at(mean, inv, pts[:, :3])
    mean /= np.maximum(counts, 1)[:, None]
    cx, cy = x0 + (ix + 0.5) * vx, y0 + (iy + 0.5) * vy
    f = np.column_stack(
        [pts[:, :4], pts[:, :3] - mean[inv], pts[:, 0] - cx, pts[:, 1] - cy, pts[:, 2] - (z0 + vz / 2)]
    )
    feats[inv, rank] = f
    coords = np.zeros((p, 2), np.int32)
    coords[inv] = np.column_stack([iy, ix])
    return feats, coords, counts.astype(np.int32)


def make_anchors(cfg: PillarConfig, grid_hw: tuple[int, int]) -> np.ndarray:
    """(H*W*A, 7) x, y, z, l, w, h, yaw; layout (H, W, class, rotation)."""
    h, w = grid_hw
    x0, y0, _, x1, y1, _ = cfg.pc_range
    xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
    ys = y0 + (np.arange(h) + 0.5) * (y1 - y0) / h
    per_cell = np.array([[l_, w_, h_, zc, r] for (l_, w_, h_, zc) in cfg.anchors for r in cfg.rotations])
    gx, gy = np.meshgrid(xs, ys)
    n = len(per_cell)
    out = np.empty((h, w, n, 7))
    out[..., 0], out[..., 1] = gx[..., None], gy[..., None]
    out[..., 2] = per_cell[:, 3]
    out[..., 3:6] = per_cell[:, :3]
    out[..., 6] = per_cell[:, 4]
    return out.reshape(-1, 7)


def encode_boxes(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    xa, ya, za, la, wa, ha, ra = anchors.T
    xg, yg, zg, lg, wg, hg, rg = boxes.T
    diag = np.hypot(la, wa)
    return np.column_stack(
        [
            (xg - xa) / diag,
            (yg - ya) / diag,
            (zg + hg / 2 - za - ha / 2) / ha,
            np.log(lg / la),
            np.log(wg / wa),
            np.log(hg / ha),
            rg - ra,
        ]
    )


def decode_boxes(deltas: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    xa, ya, za, la, wa, ha, ra = anchors.T
    xt, yt, zt, lt, wt, ht, rt = deltas.T
    diag = np.hypot(la, wa)
    hg = np.exp(ht) * ha
    return np.column_stack(
        [
            xt * diag + xa,
            yt * diag + ya,
            zt * ha + za + ha / 2 - hg / 2,
            np.exp(lt) * la,
            np.exp(wt) * wa,
            hg,
            rt + ra,
        ]
    )


def postprocess(
    cls_logits: np.ndarray,
    box_deltas: np.ndarray,
    dir_logits: np.ndarray,
    anchors: np.ndarray,
    cfg: PillarConfig,
) -> list[Box3D]:
    scores_all = 1.0 / (1.0 + np.exp(-np.asarray(cls_logits, np.float64)))
    label, score = scores_all.argmax(1), scores_all.max(1)
    idx = np.flatnonzero(score > cfg.score_thresh)
    idx = idx[np.argsort(-score[idx], kind="stable")[:_NMS_PRE]]
    if len(idx) == 0:
        return []
    boxes = decode_boxes(np.asarray(box_deltas, np.float64)[idx], np.asarray(anchors, np.float64)[idx])
    period = np.pi  # two direction bins
    rot = boxes[:, 6] - _DIR_OFFSET
    rot = rot - np.floor(rot / period) * period
    boxes[:, 6] = rot + _DIR_OFFSET + period * np.asarray(dir_logits)[idx].argmax(1)
    out: list[Box3D] = []
    for b, lab, s in zip(boxes, label[idx], score[idx]):
        cand = Box3D(
            *map(float, b[:3]), *map(float, b[3:6]), float(wrap_angle(b[6])), _CLASSES[lab], float(s)
        )
        if all(iou_bev(cand, k) <= cfg.nms_iou for k in out):
            out.append(cand)
            if len(out) == cfg.max_det:
                break
    return out


@lru_cache(maxsize=4)
def _anchors_for(cfg: PillarConfig) -> np.ndarray:
    ny, nx = cfg.canvas_hw
    return make_anchors(cfg, grid_hw=(ny // 2, nx // 2))  # first backbone stride is 2


def detect(
    points: np.ndarray,
    cfg: PillarConfig,
    network: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[Box3D]:
    """Shared by the torch and TensorRT detectors; only `network` differs. No pillars -> no network call."""
    feats, coords, num = voxelize(points, cfg)
    if len(feats) == 0:
        return []
    cls, box, d = network(feats, coords, num)
    return postprocess(cls, box, d, _anchors_for(cfg), cfg)


def _cbr(conv: nn.Module, c_out: int) -> list[nn.Module]:
    return [conv, nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01), nn.ReLU()]


class _PFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 64, bias=False)
        self.norm = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (P, N, 10) -> (P, 64)
        p, n, _ = x.shape
        x = torch.relu(self.norm(self.linear(x).reshape(p * n, 64))).reshape(p, n, 64)
        return x.max(dim=1)[0]


class _VFE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pfn_layers = nn.ModuleList([_PFN()])


class _Backbone(nn.Module):
    """OpenPCDet BaseBEVBackbone: layers [3, 5, 5], strides [2, 2, 2], filters [64, 128, 256], upsample [1, 2, 4]."""

    def __init__(self, c_in: int = 64) -> None:
        super().__init__()
        nums, strides, filters, ups = (3, 5, 5), (2, 2, 2), (64, 128, 256), (1, 2, 4)
        self.blocks, self.deblocks = nn.ModuleList(), nn.ModuleList()
        for i, c in enumerate(filters):
            layers: list[nn.Module] = [nn.ZeroPad2d(1)]
            layers += _cbr(nn.Conv2d(c_in if i == 0 else filters[i - 1], c, 3, strides[i], bias=False), c)
            for _ in range(nums[i]):
                layers += _cbr(nn.Conv2d(c, c, 3, padding=1, bias=False), c)
            self.blocks.append(nn.Sequential(*layers))
            up = nn.ConvTranspose2d(c, 128, ups[i], ups[i], bias=False)
            self.deblocks.append(nn.Sequential(*_cbr(up, 128)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for blk, deb in zip(self.blocks, self.deblocks):
            x = blk(x)
            outs.append(deb(x))
        return torch.cat(outs, dim=1)


class _Head(nn.Module):
    def __init__(self, n_anchor: int, n_cls: int) -> None:
        super().__init__()
        self.conv_cls = nn.Conv2d(384, n_anchor * n_cls, 1)
        self.conv_box = nn.Conv2d(384, n_anchor * 7, 1)
        self.conv_dir_cls = nn.Conv2d(384, n_anchor * 2, 1)


def _rows(t: torch.Tensor, c: int) -> torch.Tensor:  # (1, A*c, H, W) -> (H*W*A, c), rows (H, W, anchor)
    return t.permute(0, 2, 3, 1).reshape(-1, c)


class PointPillarsNet(nn.Module):
    """Minimal PointPillars forward; submodule names equal OpenPCDet's so its state dict loads unchanged."""

    def __init__(self, pillar_cfg: PillarConfig, num_classes: int = 3) -> None:
        super().__init__()
        self.canvas = pillar_cfg.canvas_hw
        self.num_classes = num_classes
        self.vfe = _VFE()
        self.backbone_2d = _Backbone()
        self.dense_head = _Head(len(pillar_cfg.anchors) * len(pillar_cfg.rotations), num_classes)

    def forward(self, feats: torch.Tensor, coords: torch.Tensor, num_points: torch.Tensor):
        mask = torch.arange(feats.shape[1], device=feats.device)[None, :] < num_points[:, None]
        pillar = self.vfe.pfn_layers[0](feats * mask[..., None].to(feats.dtype))  # (P, 64)
        ny, nx = self.canvas
        idx = coords[:, 0].long() * nx + coords[:, 1].long()
        canvas = pillar.new_zeros(ny * nx, 64).index_put((idx,), pillar)  # ScatterND on export
        bev = self.backbone_2d(canvas.t().reshape(1, 64, ny, nx))
        h = self.dense_head
        return (
            _rows(h.conv_cls(bev), self.num_classes),
            _rows(h.conv_box(bev), 7),
            _rows(h.conv_dir_cls(bev), 2),
        )


def load_openpcdet_state_dict(net: PointPillarsNet, ckpt_path: Path) -> None:
    """Load an OpenPCDet checkpoint ({'model_state': ...} or a bare state dict); strict on our submodules.

    Unverified until Phase G3: key names are assumed to be vfe.pfn_layers.0.*, backbone_2d.{blocks,deblocks}.*,
    dense_head.{conv_cls,conv_box,conv_dir_cls}.*.
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model_state", ck)
    net.load_state_dict(
        {k: v for k, v in sd.items() if k.split(".")[0] in ("vfe", "backbone_2d", "dense_head")}
    )
