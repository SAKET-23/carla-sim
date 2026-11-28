import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "identify_vehicle", Path(__file__).resolve().parents[2] / "scripts/identify_vehicle.py"
)
iv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(iv)

L = 3.005


@pytest.mark.parametrize("v", iv.STEP_SPEEDS)
def test_step_deltas_hit_target_lateral_accelerations(v):
    d = iv.step_deltas(L, v)
    assert sorted(abs(x) for x in d) == pytest.approx(
        sorted(a * L / v**2 for a in iv.STEP_ALATS for _ in (0, 1))
    )
    assert max(abs(x) * v**2 / L for x in d) == pytest.approx(3.0)
    assert sum(d) == pytest.approx(0.0)  # both signs


def test_obstacle_layers_unloaded_then_reloaded(fake_carla):
    sim = SimpleNamespace(world=fake_carla.FakeWorld())
    ml = fake_carla.MapLayer
    want = [ml.Buildings, ml.Props, ml.StreetLights, ml.Walls, ml.Foliage, ml.ParkedVehicles]
    with iv.obstacles_unloaded(sim):
        assert sim.world.unloaded_layers == want
        assert all(isinstance(x, ml) for x in sim.world.unloaded_layers)
        assert not hasattr(sim.world, "loaded_layers")
    assert sim.world.loaded_layers == want
    assert ml.Ground not in sim.world.loaded_layers
