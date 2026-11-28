"""Generate the segmentation dataset (Plan/03-models/segmentation.md §3): RGB and semantic camera, same pose and frame.

gen_seg_dataset.py --towns Town01,Town03_Opt,Town04,Town05 --frames-per-town 1500 --split train --out DIR --quality Low
(then `--towns Town02 --frames-per-town 800 --split val`, `--towns Town10HD --frames-per-town 1000 --split test`).
The server's quality level is not readable through the API: start_carla.sh writes it to
<quality-dir>/carla-<port>.quality, and this script exits 2 if that file is missing or differs from --quality (ECC-26).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from hybrid_av.perception.datasets.carla_seg_writer import SegWriter
from hybrid_av.sim.carla_backend.client import CarlaSimConfig, carla_session
from hybrid_av.sim.carla_backend.traffic import open_collector

RIG = Path(__file__).resolve().parents[1] / "configs/sensors/carla_rig.yaml"


def check_quality(quality: str, directory: Path, port: int) -> None:
    f = directory / f"carla-{port}.quality"
    got = f.read_text().strip() if f.exists() else None
    if got != quality:
        print(f"refusing to run: {f} says {got!r}, --quality is {quality!r} (ECC-26)", file=sys.stderr)
        sys.exit(2)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--towns", required=True, help="comma-separated")
    ap.add_argument("--frames-per-town", type=int, required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quality", required=True, choices=["Low", "Epic"])
    ap.add_argument("--quality-dir", type=Path, default=Path("/workspace/logs"))
    ap.add_argument("--stride", type=int, default=10, help="save every Nth tick (0.5 s)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--vehicles", type=int, default=30)
    ap.add_argument("--walkers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    a = ap.parse_args(argv)
    check_quality(a.quality, a.quality_dir, a.port)
    writer = SegWriter(a.out)
    idx0 = len(list((a.out / "images").glob("*.png")))  # val/test runs append to the same directory
    n = 0
    for town in a.towns.split(","):
        with carla_session(CarlaSimConfig(host=a.host, port=a.port, town=town, seed=a.seed)) as session:
            c = open_collector(
                session, RIG, ["camera_front", "camera_front_semantic"], a.seed, a.vehicles, a.walkers
            )
            saved = 0
            for k in range(a.warmup + a.frames_per_town * a.stride):
                _, data = c.tick()
                rgb_d, sem_d = data["camera_front"], data["camera_front_semantic"]
                if k < a.warmup or (k - a.warmup) % a.stride or rgb_d is None or sem_d is None:
                    continue
                if saved % 100 == 0:
                    c.set_weather(saved // 100)
                bgra = np.frombuffer(rgb_d.raw_data, np.uint8).reshape(rgb_d.height, rgb_d.width, 4)
                tags = np.frombuffer(sem_d.raw_data, np.uint8).reshape(sem_d.height, sem_d.width, 4)[
                    :, :, 2
                ]  # ECC-31
                writer.write_pair(idx0 + n, bgra[:, :, 2::-1], tags)
                n += 1
                saved += 1
        print(f"{town}: {saved} frames")
    (a.out / f"{a.split}.txt").write_text("".join(f"{i:06d}\n" for i in range(idx0, idx0 + n)))


if __name__ == "__main__":
    main()
