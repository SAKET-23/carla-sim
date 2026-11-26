import socket

import pytest


def _carla_up(host="localhost", port=2000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    # colcon test has no addopts filter: skip the live tests unless a CARLA server answers
    if not _carla_up():
        skip = pytest.mark.skip(reason="no CARLA server on localhost:2000")
        for item in items:
            if item.get_closest_marker("carla"):
                item.add_marker(skip)
