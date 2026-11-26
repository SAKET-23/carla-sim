import numpy as np
import pytest
from PIL import Image

from hybrid_av.perception.base import Box3D
from hybrid_av.perception.classes import CARLA_TAG_TO_TRAIN
from hybrid_av.perception.datasets.carla_kitti_writer import DetectionWriter, read_labels
from hybrid_av.perception.datasets.carla_seg_writer import SegWriter
from hybrid_av.types import ObjectClass


def test_label_roundtrip_keeps_lidar_frame(tmp_path):
    boxes = [
        Box3D(x=15.0, y=-3.0, z=-0.9, length=4.5, width=1.9, height=1.6, yaw=0.3, cls=ObjectClass.CAR, score=1.0),
        Box3D(x=8.0, y=2.0, z=-0.8, length=0.6, width=0.6, height=1.8, yaw=-2.9, cls=ObjectClass.PEDESTRIAN, score=1.0),
    ]  # fmt: skip
    DetectionWriter(tmp_path).write_frame(7, np.ones((5, 4), np.float32), boxes)
    back = read_labels(tmp_path / "labels/000007.txt")
    for a, b in zip(boxes, back):
        assert (a.cls, a.x, a.y, a.z, a.length, a.width, a.height) == pytest.approx(
            (b.cls, b.x, b.y, b.z, b.length, b.width, b.height), abs=1e-4
        )
        assert a.yaw == pytest.approx(b.yaw, abs=1e-4)


def test_label_line_format_and_truck_is_car(tmp_path):
    DetectionWriter(tmp_path).write_frame(
        0,
        np.zeros((1, 4), np.float32),
        [
            Box3D(
                x=10.0,
                y=-1.0,
                z=0.2,
                length=8.0,
                width=2.5,
                height=3.0,
                yaw=0.5,
                cls=ObjectClass.TRUCK,
                score=1.0,
            )
        ],
    )
    f = (tmp_path / "labels/000000.txt").read_text().split()
    assert (
        len(f) == 8
        and f[7] == "Car"
        and list(map(float, f[:7])) == pytest.approx([10.0, -1.0, 0.2, 8.0, 2.5, 3.0, 0.5])
    )


def test_points_and_split_files(tmp_path):
    pts = np.arange(8, dtype=np.float32).reshape(2, 4)
    w = DetectionWriter(tmp_path)
    w.write_frame(3, pts, [])
    w.write_split("train", [3])
    np.testing.assert_array_equal(np.load(tmp_path / "points/000003.npy"), pts)
    assert (tmp_path / "labels/000003.txt").read_text() == ""
    assert (tmp_path / "ImageSets/train.txt").read_text().strip() == "000003"


def test_seg_pair(tmp_path):
    tags = np.array([[1, 24], [14, 200]], np.uint8)
    SegWriter(tmp_path).write_pair(1, np.zeros((2, 2, 3), np.uint8), tags)
    lab = np.array(Image.open(tmp_path / "labels/000001.png"))
    assert lab.tolist() == [[CARLA_TAG_TO_TRAIN[1], CARLA_TAG_TO_TRAIN[24]], [CARLA_TAG_TO_TRAIN[14], 255]]
