import numpy as np

from hybrid_av.perception.classes import (
    CARLA_ACTOR_TO_CLASS,
    CARLA_TAG_TO_TRAIN,
    CARLA_TAGS,
    TRAIN_CLASSES,
    drivable_ratio,
)
from hybrid_av.types import ObjectClass


def test_every_carla_tag_maps_to_valid_train_id_or_ignore():
    for tag in CARLA_TAGS:
        assert CARLA_TAG_TO_TRAIN[tag] < len(TRAIN_CLASSES) or CARLA_TAG_TO_TRAIN[tag] == 255


def test_table_matches_segmentation_md():
    assert CARLA_TAG_TO_TRAIN[1] == CARLA_TAG_TO_TRAIN[24] == 0  # road
    assert CARLA_TAG_TO_TRAIN[14] == CARLA_TAG_TO_TRAIN[19] == 2  # vehicle
    assert CARLA_TAG_TO_TRAIN[12] == CARLA_TAG_TO_TRAIN[13] == 3  # person
    assert CARLA_TAG_TO_TRAIN[11] == 6 and CARLA_TAG_TO_TRAIN[0] == 255 and CARLA_TAG_TO_TRAIN[200] == 255
    assert len(TRAIN_CLASSES) == 8


def test_actor_classes():
    assert CARLA_ACTOR_TO_CLASS("vehicle.tesla.model3") == ObjectClass.CAR
    assert CARLA_ACTOR_TO_CLASS("vehicle.carlamotors.carlacola") == ObjectClass.TRUCK
    assert CARLA_ACTOR_TO_CLASS("vehicle.diamondback.century") == ObjectClass.CYCLIST
    assert CARLA_ACTOR_TO_CLASS("walker.pedestrian.0001") == ObjectClass.PEDESTRIAN


def test_drivable_ratio_roi():
    road = np.zeros((100, 100), np.uint8)
    assert drivable_ratio(road) == 1.0
    assert drivable_ratio(np.full((100, 100), 255, np.uint8)) == 0.0
    top_only = np.full((100, 100), 255, np.uint8)
    top_only[:60] = 0  # road only above the ROI
    assert drivable_ratio(top_only) == 0.0
