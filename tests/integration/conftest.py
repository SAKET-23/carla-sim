"""Fixtures for the live CARLA suite (Plan/06-integration/integration-check.md). No module here imports `carla` at
import time: the Mac has no wheel."""

import os
import socket
from pathlib import Path

import pytest
import torch
import yaml

from hybrid_av.policy.io import save_policy
from hybrid_av.policy.network import CompactPolicy

FIXTURE = Path(__file__).parent / "fixtures/town03_points.yaml"


def _server_up(host="localhost", port=None, timeout=2.0) -> bool:
    port = port or int(os.environ.get("CARLA_PORT", "2000"))
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def carla_server():
    if not _server_up():
        pytest.skip("no CARLA server on localhost:2000")


@pytest.fixture(scope="session")
def points(carla_server):
    if not FIXTURE.exists():
        pytest.skip("run find_map_features.py --fixture first")
    return yaml.safe_load(FIXTURE.read_text())


@pytest.fixture
def make_sim(carla_server):
    """Factory for CarlaSim; every sim is closed at teardown, so the server is left asynchronous."""
    from hybrid_av.sim.carla_backend.client import CarlaSimConfig
    from hybrid_av.sim.carla_backend.sim import CarlaSim

    sims = []

    def make(**kw):
        sim = CarlaSim(CarlaSimConfig(**{"town": "Town03_Opt", "sensors": None, **kw}))
        sims.append(sim)
        return sim

    yield make
    for sim in sims:
        sim.close()


@pytest.fixture
def seeded_compact_policy_path(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "seeded_policy.pt"
    save_policy(CompactPolicy(), path, {"name": "seeded"})
    return path
