import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hybrid_av.perception.base import Box3D
from hybrid_av.types import ObjectClass

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_points_in_box_counts_rotated_box():
    gen = _load("gen_detection_dataset")
    b = Box3D(10.0, 0.0, 0.8, 4.0, 2.0, 1.6, 0.0, ObjectClass.CAR, 1.0)
    pts = np.array([[10, 0, 0.8, 1], [11.9, 0.9, 0.8, 1], [12.1, 0, 0.8, 1], [10, 0, 2.0, 1]], np.float32)
    assert gen.points_in_box(pts, b) == 2
    turned = Box3D(10.0, 0.0, 0.8, 4.0, 2.0, 1.6, np.pi / 2, ObjectClass.CAR, 1.0)  # long side along y
    assert gen.points_in_box(np.array([[10, 1.9, 0.8, 1], [11.9, 0, 0.8, 1]], np.float32), turned) == 1


def test_label_boxes_need_points_and_range(fake_carla):
    gen = _load("gen_detection_dataset")
    world = fake_carla.last_world
    ego = world.spawn_actor(fake_carla.ActorBlueprint("vehicle.tesla.model3"), fake_carla.Transform())
    near = world.spawn_actor(
        fake_carla.ActorBlueprint("vehicle.audi.tt"),
        fake_carla.Transform(fake_carla.Location(10.0, 0.0, 0.0)),
    )
    world.spawn_actor(
        fake_carla.ActorBlueprint("walker.pedestrian.0001"),
        fake_carla.Transform(fake_carla.Location(30.0, 0.0, 0.0)),
    )
    near.bounding_box = fake_carla.BoundingBox(
        fake_carla.Location(0, 0, 0.75), fake_carla.Vector3D(2.0, 1.0, 0.75)
    )
    pts = np.tile([10.0, 0.0, 0.75, 1.0], (12, 1)).astype(
        np.float32
    )  # 12 points in the car, none in the walker
    boxes = gen.label_boxes(world, ego, world.get_snapshot(), pts)
    assert [b.cls for b in boxes] == [ObjectClass.CAR]
    assert boxes[0].x == pytest.approx(10.0) and boxes[0].length == pytest.approx(4.0)


def test_detection_script_layout(fake_carla, tmp_path):
    _load("gen_detection_dataset").main(
        ["--towns", "Town03_Opt", "--frames-per-town", "2", "--split", "train", "--out", str(tmp_path),
         "--stride", "10", "--warmup", "0", "--vehicles", "3", "--walkers", "2", "--render-check", "1"]
    )  # fmt: skip
    assert len(list((tmp_path / "points").glob("*.npy"))) == 2
    assert len(list((tmp_path / "labels").glob("*.txt"))) == 2
    assert (tmp_path / "ImageSets/train.txt").read_text().split() == ["000000", "000001"]
    assert len(list((tmp_path / "render_check").glob("*.png"))) == 1
    assert not fake_carla.last_world.get_settings().synchronous_mode  # restored


def test_detection_script_appends_val_split(fake_carla, tmp_path):
    gen = _load("gen_detection_dataset")
    base = ["--stride", "10", "--warmup", "0", "--vehicles", "1", "--walkers", "0", "--out", str(tmp_path)]
    gen.main(["--towns", "Town03_Opt", "--frames-per-town", "1", "--split", "train", *base])
    gen.main(["--towns", "Town02", "--frames-per-town", "1", "--split", "val", *base])
    assert (tmp_path / "ImageSets/val.txt").read_text().split() == ["000001"]


def test_seg_script_layout(fake_carla, tmp_path):
    q = tmp_path / "logs"
    q.mkdir()
    (q / "carla-2000.quality").write_text("Low\n")
    _load("gen_seg_dataset").main(
        ["--towns", "Town03_Opt", "--frames-per-town", "2", "--split", "train", "--out", str(tmp_path / "seg"),
         "--quality", "Low", "--quality-dir", str(q), "--stride", "10", "--warmup", "0", "--vehicles", "2", "--walkers", "1"]
    )  # fmt: skip
    assert len(list((tmp_path / "seg/images").glob("*.png"))) == 2
    assert len(list((tmp_path / "seg/labels").glob("*.png"))) == 2
    assert not fake_carla.last_world.get_settings().synchronous_mode


@pytest.mark.parametrize("content", ["Epic\n", None])
def test_seg_script_refuses_quality_mismatch(fake_carla, tmp_path, content):
    if content:
        (tmp_path / "carla-2000.quality").write_text(content)
    with pytest.raises(SystemExit) as e:
        _load("gen_seg_dataset").main(
            ["--towns", "Town03_Opt", "--frames-per-town", "1", "--split", "train", "--out", str(tmp_path / "seg"),
             "--quality", "Low", "--quality-dir", str(tmp_path)]
        )  # fmt: skip
    assert e.value.code == 2
    assert fake_carla.load_world_calls == 0  # refused before touching the server


def test_traffic_is_destroyed_with_controllers_stopped(fake_carla, tmp_path):
    _load("gen_detection_dataset").main(
        ["--towns", "Town03_Opt", "--frames-per-town", "1", "--split", "train", "--out", str(tmp_path),
         "--warmup", "0", "--vehicles", "2", "--walkers", "2"]
    )  # fmt: skip
    assert not fake_carla.last_world.actors
