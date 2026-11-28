"""IC-01...IC-04: server, settings, lifecycle (integration-check.md section A)."""

import itertools
import time

import pytest

from hybrid_av.sim.carla_backend.client import CarlaSimConfig, carla_session
from hybrid_av.sim.carla_backend.routes import plain_scenario, straight_route

pytestmark = pytest.mark.carla


def test_versions(carla_server):
    """IC-01 — see Plan/06-integration/integration-check.md; likely fix: ECC-01."""
    import carla

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    assert client.get_server_version() == client.get_client_version() == "0.9.16"


def test_sync_settings_applied(carla_server):
    """IC-02 — see Plan/06-integration/integration-check.md; likely fix: ECC-02."""
    import carla

    with carla_session(CarlaSimConfig()) as s:
        st = s.world.get_settings()
        assert st.synchronous_mode and st.fixed_delta_seconds == pytest.approx(0.05)
        assert st.max_substep_delta_time == pytest.approx(0.01) and st.max_substeps == 10
        # the API has no Traffic Manager getter: an autopilot vehicle ticking 200 times without a client timeout
        # shows the TM is synchronous too (a mismatch makes world.tick() time out)
        wmap = s.world.get_map()
        car = s.world.spawn_actor(
            s.world.get_blueprint_library().find("vehicle.tesla.model3"), wmap.get_spawn_points()[0]
        )
        s.actors.append(car)
        car.set_autopilot(True, s.tm.get_port())
        frames, times = [], []
        for _ in range(200):
            s.world.tick()
            snap = s.world.get_snapshot()
            frames.append(snap.frame)
            times.append(snap.timestamp.elapsed_seconds)
        assert all(b - a == 1 for a, b in itertools.pairwise(frames))
        assert all(abs((b - a) - 0.05) < 1e-6 for a, b in itertools.pairwise(times))
    assert not carla.Client("localhost", 2000).get_world().get_settings().synchronous_mode


def test_close_restores_async_and_destroys_actors(make_sim, points):
    """IC-03 — see Plan/06-integration/integration-check.md; likely fix: ECC-03."""
    import carla

    world = carla.Client("localhost", 2000).get_world()
    before = len(world.get_actors())
    rc = straight_route(points, "Town03_Opt")
    for _ in range(2):  # a second reset() -> close() cycle leaves the actor count where it started
        sim = make_sim(sensors=None)
        sim.reset(plain_scenario(rc))
        sim.close()
        assert not world.get_settings().synchronous_mode
        assert len(world.get_actors()) == before
    # shortcut: the spec's role_name check is replaced by the actor count, CarlaSim sets no role_name


def test_reset_time(make_sim, points):
    """IC-04 — see Plan/06-integration/integration-check.md; likely fix: ECC-03."""
    sim = make_sim(sensors=None)
    sc = plain_scenario(straight_route(points, "Town03_Opt"))
    sim.reset(sc)  # loads the town
    t0 = time.monotonic()
    obs = sim.reset(sc)
    assert time.monotonic() - t0 < 10.0
    assert abs(obs.ego.vx) < 0.05 and abs(obs.ego.r) < 0.01
