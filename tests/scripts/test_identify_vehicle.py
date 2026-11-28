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


def test_run_long_settles_past_reset_spike_and_keeps_short_runs(monkeypatch):
    from hybrid_av.types import EgoState

    class Sim:
        dt = 0.05

        def reset(self, sc):
            self.v, self.ticks = 10.0, 0
            return self._obs(400.0)  # reset spike

        def _obs(self, ax):
            return SimpleNamespace(ego=EgoState(0, 0, 0, 0, self.v, 0, 0, ax))

        def step(self, cmd):
            self.ticks += 1
            self.v += 0.05 * (6.0 * cmd.throttle - 8.0 * cmd.brake - 0.1 * self.v)
            return self._obs(0.1), SimpleNamespace(route_completed=False)

    monkeypatch.setattr(iv, "plain_scenario", lambda rc, v0: None)
    sim = Sim()
    vx, ax, _t, brk = iv.run_long(sim, None, 10.0, 12.0, lambda e: (0.0, 1.0), lambda e: e.vx < 0.2)
    assert len(vx) > 0 and ax.max() < 50 and brk[0] == 1.0  # recorded from the open-loop start, no spike
    assert sim.ticks > 30 and vx[0] == pytest.approx(10.0, abs=1.0)
