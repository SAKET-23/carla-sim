"""Multi-object tracker: world-frame CV EKF, M-of-N lifecycle, LiDAR boxes plus radar range rate."""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hybrid_av.geometry import base_to_world, wrap_angle
from hybrid_av.perception.base import Box3D, RadarDetection
from hybrid_av.tracking.association import assign, mahalanobis_cost
from hybrid_av.tracking.ekf import cv_predict, update_position, update_radar
from hybrid_av.tracking.radar import RadarExtrinsic, compensated_range_rate, detection_to_base
from hybrid_av.types import EgoState, ObjectClass, ObjectState

__all__ = ["RadarExtrinsic", "Tracker", "TrackerConfig"]

_INCOMPATIBLE = 1e9


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    q_accel: float = 2.0
    q_accel_pedestrian: float = 1.0
    sigma_lidar_pos: float = 0.3
    sigma_radar_rr: float = 0.25
    init_sigma_pos: float = 0.5
    init_sigma_vel: float = 5.0
    gate_chi2: float = 9.21
    radar_gate_m: float = 2.5
    radar_max_altitude_deg: float = 5.0
    radar_min_height: float = 0.2
    radar_max_height: float = 2.5
    min_score: float = 0.4
    confirm_m: int = 3
    confirm_n: int = 5
    max_misses: int = 5
    max_misses_tentative: int = 2
    alpha_size: float = 0.3
    max_range: float = 70.0
    class_compat: tuple[tuple[str, str], ...] = (("CAR", "TRUCK"),)


class _Track:
    def __init__(self, tid: int, box: Box3D, xy: np.ndarray, yaw: float, cfg: TrackerConfig):
        self.id, self.cls = tid, box.cls
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([cfg.init_sigma_pos**2] * 2 + [cfg.init_sigma_vel**2] * 2)
        self.z, self.yaw = box.z, yaw
        self.size = np.array([box.length, box.width, box.height])
        self.hits: deque[bool] = deque([True], maxlen=cfg.confirm_n)
        self.misses = 0
        self.confirmed = cfg.confirm_m <= 1

    def blend(self, box: Box3D, yaw: float, a: float) -> None:
        self.z += a * (box.z - self.z)
        self.size += a * (np.array([box.length, box.width, box.height]) - self.size)
        d = wrap_angle(yaw - self.yaw)
        if abs(d) > math.pi / 2:  # boxes are symmetric: use the closer of the two headings
            d = wrap_angle(d + math.pi)
        self.yaw = wrap_angle(self.yaw + a * d)


class Tracker:
    def __init__(self, cfg: TrackerConfig, radars: Sequence[RadarExtrinsic]) -> None:
        self.cfg = cfg
        self._radars = {r.sensor_id: r for r in radars}
        self._compat = {frozenset(p) for p in cfg.class_compat}
        self.reset()

    def reset(self) -> None:
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._t: float | None = None

    def predict(self, t: float) -> None:
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        if dt <= 0:
            return
        for tr in self._tracks:
            q = self.cfg.q_accel_pedestrian if tr.cls == ObjectClass.PEDESTRIAN else self.cfg.q_accel
            tr.x, tr.P = cv_predict(tr.x, tr.P, dt, q)

    def _compatible(self, a: ObjectClass, b: ObjectClass) -> bool:
        return a == b or frozenset((a.name, b.name)) in self._compat

    def update_boxes(self, t: float, boxes: Sequence[Box3D], ego: EgoState) -> None:
        cfg = self.cfg
        boxes = [b for b in boxes if math.hypot(b.x, b.y) <= cfg.max_range]
        xy = base_to_world(np.array([[b.x, b.y] for b in boxes]).reshape(-1, 2), ego)
        yaws = [b.yaw + ego.psi for b in boxes]

        R = np.eye(2) * cfg.sigma_lidar_pos**2
        cost = mahalanobis_cost(
            np.array([tr.x[:2] for tr in self._tracks]).reshape(-1, 2),
            np.array([tr.P[:2, :2] for tr in self._tracks]).reshape(-1, 2, 2),
            xy,
            np.broadcast_to(R, (len(boxes), 2, 2)),
        )
        for i, tr in enumerate(self._tracks):
            for j, b in enumerate(boxes):
                if not self._compatible(tr.cls, b.cls):
                    cost[i, j] = _INCOMPATIBLE
        matches, unmatched_tracks, unmatched_dets = assign(cost, cfg.gate_chi2)

        for i, j in matches:
            tr = self._tracks[i]
            tr.x, tr.P, _ = update_position(tr.x, tr.P, xy[j], R)
            tr.blend(boxes[j], yaws[j], cfg.alpha_size)
            tr.hits.append(True)
            tr.misses = 0
            tr.confirmed = tr.confirmed or sum(tr.hits) >= cfg.confirm_m
        for i in unmatched_tracks:
            tr = self._tracks[i]
            tr.hits.append(False)
            tr.misses += 1
        self._tracks = [
            tr
            for tr in self._tracks
            if tr.misses < (cfg.max_misses if tr.confirmed else cfg.max_misses_tentative)
        ]
        for j in unmatched_dets:
            if boxes[j].score >= cfg.min_score:
                self._tracks.append(_Track(self._next_id, boxes[j], xy[j], yaws[j], cfg))
                self._next_id += 1

    def update_radar(self, t: float, dets: Sequence[RadarDetection], ego: EgoState) -> None:
        cfg = self.cfg
        max_alt = math.radians(cfg.radar_max_altitude_deg)
        for det in dets:
            ext = self._radars.get(det.sensor_id)
            if ext is None or abs(det.altitude) > max_alt or not self._tracks:
                continue
            p = detection_to_base(det, ext)
            if not cfg.radar_min_height <= p[2] <= cfg.radar_max_height:  # base_link origin is on the ground
                continue
            pw, sw = base_to_world(np.array([p[:2], [ext.x, ext.y]]), ego)
            dists = [float(np.hypot(*(tr.x[:2] - pw))) for tr in self._tracks]
            k = int(np.argmin(dists))
            if dists[k] > cfg.radar_gate_m:
                continue
            tr = self._tracks[k]
            try:
                tr.x, tr.P, _ = update_radar(
                    tr.x, tr.P, compensated_range_rate(det, ext, ego), sw, cfg.sigma_radar_rr**2
                )
            except ValueError:  # track within 0.5 m of the radar cannot be associated
                continue

    def tracks(self, confirmed_only: bool = True) -> list[ObjectState]:
        return [
            ObjectState(
                id=tr.id,
                cls=tr.cls,
                x=float(tr.x[0]),
                y=float(tr.x[1]),
                z=tr.z,
                yaw=tr.yaw,
                length=float(tr.size[0]),
                width=float(tr.size[1]),
                height=float(tr.size[2]),
                vx=float(tr.x[2]),
                vy=float(tr.x[3]),
                pos_cov=tr.P[:2, :2].copy(),
            )
            for tr in self._tracks
            if tr.confirmed or not confirmed_only
        ]
