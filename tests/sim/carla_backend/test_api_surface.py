from pathlib import Path

import pytest

from tests.sim.carla_backend.api_surface import API_SURFACE, used_names

_ROOT = Path(__file__).resolve().parents[3]
BACKEND = [
    _ROOT / "src/hybrid_av/sim/carla_backend" / f
    for f in ("client.py", "sensors.py", "scenario_setup.py", "sim.py", "traffic.py")
]


def _used():
    return used_names([p for p in BACKEND if p.exists()])


def test_fake_covers_backend_api_surface():
    import tests.fakes.carla_fake as fake

    missing = {n for n in _used() if not _resolves(fake, n)}
    assert not missing, f"backend uses CARLA names the fake lacks: {sorted(missing)}"


def test_fake_covers_documented_surface():
    import tests.fakes.carla_fake as fake

    assert not {n for n in API_SURFACE if not _resolves(fake, n)}


def test_surface_list_is_complete():
    assert _used() <= API_SURFACE


def test_scanner_maps_names_to_classes(tmp_path):
    src = tmp_path / "x.py"
    src.write_text(
        "import carla\n"
        "def f(world, ego, sensor):\n"
        "    world.tick(); ego.apply_control(carla.VehicleControl()); sensor.stop()\n"
        "    return carla.command.DestroyActor(1), carla.LaneType.Driving\n"
    )
    assert used_names([src]) == {
        "World.tick", "Vehicle.apply_control", "Sensor.stop", "VehicleControl",
        "command.DestroyActor", "LaneType.Driving",
    }  # fmt: skip


@pytest.mark.carla
def test_real_carla_has_api_surface():
    import carla

    assert not {n for n in API_SURFACE if not _resolves(carla, n)}


def _resolves(mod, dotted):
    obj = mod
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True
