import socket
import sys

import pytest


def _carla_up(host="localhost", port=2000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    # an explicit -m replaces addopts, so skip carla tests here unless a server answers
    if any(i.get_closest_marker("carla") for i in items) and not _carla_up():
        skip = pytest.mark.skip(reason="no CARLA server on localhost:2000")
        for i in items:
            if i.get_closest_marker("carla"):
                i.add_marker(skip)


@pytest.fixture
def fake_carla(monkeypatch):
    """Install the fake as `carla` (and the CARLA `agents` package) for one test; returns the fake module."""
    import tests.fakes.carla_fake as fake
    from tests.fakes.carla_fake import agents
    from tests.fakes.carla_fake.agents import navigation
    from tests.fakes.carla_fake.agents.navigation import global_route_planner, local_planner

    fake.reset()
    for name, mod in {
        "carla": fake,
        "carla.command": fake.command,
        "agents": agents,
        "agents.navigation": navigation,
        "agents.navigation.global_route_planner": global_route_planner,
        "agents.navigation.local_planner": local_planner,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return fake
