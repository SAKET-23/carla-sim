import logging

import pytest
import yaml

from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.carla_backend.client import CarlaSimConfig
from hybrid_av.sim.carla_backend.routes import carla_scenario
from hybrid_av.sim.carla_backend.sim import CarlaSim

_open: list[CarlaSim] = []


@pytest.fixture(autouse=True)
def _close_sims(fake_carla):
    yield
    while _open:
        _open.pop().close()


def _reset(routes_yaml, tmp_path, seed, family="nominal"):
    f = tmp_path / "routes.yaml"
    f.write_text(yaml.safe_dump(routes_yaml))
    sc = carla_scenario(make_scenario(family, seed), family, seed, "Town03", f, fallback=True)
    sim = CarlaSim(CarlaSimConfig(sensors=None))
    _open.append(sim)
    sim.reset(sc)
    return sim


def test_lookup_without_fallback_still_raises(tmp_path):
    f = tmp_path / "routes.yaml"
    f.write_text("{}\n")
    with pytest.raises(LookupError, match="no CARLA route"):
        carla_scenario(make_scenario("nominal", 1), "nominal", 1, None, f)


def test_fallback_used_when_no_route(fake_carla, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        sim = _reset({}, tmp_path, seed=0)
    assert 300.0 <= sim.route().length <= 600.0
    assert "no route in carla_routes.yaml" in caplog.text


def test_configured_route_wins(fake_carla, tmp_path, caplog):
    table = {"nominal": {"town": "Town03", "start": [0, 0, 0, 0], "goal": [100, 0, 0]}}
    with caplog.at_level(logging.WARNING):
        sim = _reset(table, tmp_path, seed=0)
    assert sim.route().length == pytest.approx(100.0, abs=1.0) and not caplog.text


def test_fallback_is_deterministic_per_seed(fake_carla, tmp_path):
    def length(seed):
        return _reset({}, tmp_path, seed).route().length

    assert length(3) == pytest.approx(length(3)) and length(3) != pytest.approx(length(4))
