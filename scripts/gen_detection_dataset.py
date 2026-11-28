"""Generate the LiDAR detection dataset (Plan/03-models/detection-3d.md §2) in the OpenPCDet custom layout.

gen_detection_dataset.py --towns Town01,Town03_Opt,Town04,Town05 --frames-per-town 1600 --split train --out DIR
(then `--towns Town02 --split val`, `--towns Town10HD --split test`). Needs a CARLA server; always restores
asynchronous mode. Boxes and scan come from the same snapshot frame (ECC-30).
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hybrid_av.perception.classes import CARLA_ACTOR_TO_CLASS
from hybrid_av.perception.datasets.carla_kitti_writer import DetectionWriter
from hybrid_av.sim.carla_backend import convert
from hybrid_av.sim.carla_backend.client import CarlaSimConfig, carla_session
from hybrid_av.sim.carla_backend.traffic import open_collector
from hybrid_av.types import ObjectClass

RIG = Path(__file__).resolve().parents[1] / "configs/sensors/carla_rig.yaml"
RANGE, MIN_POINTS, BEV_M = (
    70.0,
    10,
    80.0,
)  # label range (m), LiDAR points inside the box, BEV image extent (m)


def points_in_box(pts: np.ndarray, b) -> int:
    d = pts[:, :3] - [b.x, b.y, b.z]
    c, s = math.cos(-b.yaw), math.sin(-b.yaw)
    x, y = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
    return int(np.sum((abs(x) <= b.length / 2) & (abs(y) <= b.width / 2) & (abs(d[:, 2]) <= b.height / 2)))


def lidar_in_base(datum, spec) -> np.ndarray:
    """(N, 4) library-frame points in base_link: sensor frame -> y flip (ECC-08) -> mount (yaw, then offset)."""
    pts = convert.lidar_to_lib(np.frombuffer(datum.raw_data, np.float32))
    c, s = math.cos(spec.yaw), math.sin(spec.yaw)
    out = pts.copy()
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1] + spec.x
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1] + spec.y
    out[:, 2] = pts[:, 2] + spec.z
    return out


def label_boxes(world, ego, snap, pts: np.ndarray) -> list:
    """Vehicles and walkers within 70 m with >= 10 LiDAR points inside, boxed in base_link at snapshot `snap`."""
    e = snap.find(ego.id)  # ECC-30: poses from the snapshot of the scan's frame, not from a later actor read
    state = convert.ego_from_actor(
        e.get_transform(), e.get_velocity(), e.get_angular_velocity(), e.get_acceleration(), 0.0, lambda p: p
    )
    boxes = []
    for actor in world.get_actors():
        if actor.id == ego.id or not actor.type_id.startswith(("vehicle.", "walker.")):
            continue
        at = snap.find(actor.id)
        cls = CARLA_ACTOR_TO_CLASS(actor.type_id)
        if at is None or cls == ObjectClass.UNKNOWN:
            continue
        box = actor.bounding_box
        b = convert.bbox_to_box3d(at.get_transform(), box.extent, box.location, state, cls)
        if math.hypot(b.x, b.y) <= RANGE and points_in_box(pts, b) >= MIN_POINTS:
            boxes.append(b)
    return boxes


def bev_png(pts: np.ndarray, boxes, path: Path, px: int = 800) -> None:
    """Points (white) with the boxes (green) over them, for the G3 box/scan sanity check."""
    img = Image.new("RGB", (px, px))
    scale = px / BEV_M
    to_px = lambda x, y: (px / 2 - y * scale, px / 2 - x * scale)  # forward is up, left is left
    draw = ImageDraw.Draw(img)
    draw.point(
        [to_px(x, y) for x, y in pts[:, :2] if abs(x) < BEV_M / 2 and abs(y) < BEV_M / 2],
        fill=(255, 255, 255),
    )
    for b in boxes:
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        corners = [(b.x + c * dx - s * dy, b.y + s * dx + c * dy) for dx, dy in
                   ((b.length / 2, b.width / 2), (b.length / 2, -b.width / 2), (-b.length / 2, -b.width / 2), (-b.length / 2, b.width / 2))]  # fmt: skip
        draw.polygon([to_px(x, y) for x, y in corners], outline=(0, 255, 0))
    img.save(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--towns", required=True, help="comma-separated")
    ap.add_argument("--frames-per-town", type=int, required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--render-check", type=int, default=0, help="write this many BEV PNGs with boxes over points"
    )
    ap.add_argument("--stride", type=int, default=10, help="save every Nth tick")
    ap.add_argument("--warmup", type=int, default=50, help="ticks before the first saved frame")
    ap.add_argument("--vehicles", type=int, default=30)
    ap.add_argument("--walkers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    a = ap.parse_args(argv)
    writer = DetectionWriter(a.out)
    idx0 = len(list((a.out / "points").glob("*.npy")))  # val/test runs append to the same directory
    indices, checks = [], 0
    for town in a.towns.split(","):
        with carla_session(CarlaSimConfig(host=a.host, port=a.port, town=town, seed=a.seed)) as session:
            c = open_collector(session, RIG, ["lidar"], a.seed, a.vehicles, a.walkers)
            saved = 0
            for k in range(a.warmup + a.frames_per_town * a.stride):
                snap, data = c.tick()
                if k < a.warmup or (k - a.warmup) % a.stride or data["lidar"] is None:
                    continue
                if saved % 100 == 0:
                    c.set_weather(saved // 100)
                pts = lidar_in_base(data["lidar"], c.specs[0])
                boxes = label_boxes(session.world, c.ego, snap, pts)
                idx = idx0 + len(indices)
                writer.write_frame(idx, pts, boxes)
                indices.append(idx)
                saved += 1
                if checks < a.render_check:
                    (a.out / "render_check").mkdir(exist_ok=True)
                    bev_png(pts, boxes, a.out / f"render_check/{idx:06d}.png")
                    checks += 1
        print(f"{town}: {saved} frames")
    writer.write_split(a.split, indices)


if __name__ == "__main__":
    main()
