import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

_SPEC = importlib.util.spec_from_file_location(
    "find_map_features", Path(__file__).resolve().parents[2] / "scripts" / "find_map_features.py"
)


def _load():
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def test_signalised_junction_uses_actor_filter_not_get_traffic_lights(fake_carla):
    fmf = _load()
    world = fake_carla.last_world
    assert not hasattr(world, "get_traffic_lights")  # 0.9.16 has no such method
    world.spawn_actor(fake_carla.ActorBlueprint("vehicle.audi.tt"), fake_carla.Transform())
    light = world.spawn_actor(fake_carla.ActorBlueprint("traffic.traffic_light"), fake_carla.Transform())
    behind = fake_carla.Waypoint(fake_carla.Transform(fake_carla.Location(1.0, 2.0, 3.0)))
    light.stop_waypoints = [NS(previous=lambda d: [behind])]
    assert fmf.signalised_junction(world) == [1.0, 2.0, 3.0, 0.0]


def test_curves_one_row_per_distinct_radius_and_start(fake_carla, monkeypatch):
    fmf = _load()

    def wp(x):
        return fake_carla.Waypoint(fake_carla.Transform(fake_carla.Location(x, 0.0, 0.0)))

    fwd, other = [wp(0.0)] * 12, [wp(5.0)] * 12
    monkeypatch.setattr(fmf, "lib_xy", lambda chain: None)
    monkeypatch.setattr(fmf.RouteBuilder, "from_waypoints", lambda *a: None)
    monkeypatch.setattr(fmf, "constant_radius_segments", lambda *a, **k: [(0.0, 50.0, 30.0)])
    rows = fmf.curves([fwd, fwd, other])
    assert [r["start"][0] for r in rows] == [0.0, 5.0]


def test_straight_follows_next_across_segment_boundaries(fake_carla):
    fmf = _load()

    class Wp(fake_carla.Waypoint):
        def next(self, d):
            t = self.transform
            if t.location.x >= 250.0:
                return []
            return [Wp(fake_carla.Transform(fake_carla.Location(t.location.x + d, 0.0, 0.0)))]

    # segments are 100 m each; next() carries a start on across the boundaries
    chains = [[Wp(fake_carla.Transform(fake_carla.Location(x, 0.0, 0.0)))] for x in (0.0, 100.0, 200.0)]
    assert fmf.straight(chains) == [0.0, 0.0, 0.0, 0.0]
    assert fmf.straight(chains[1:]) is None  # 150 m from x=100
