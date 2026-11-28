import pytest

from hybrid_av.sim.carla_backend.client import CarlaSimConfig, apply_sync, carla_session, restore


def test_sync_settings_applied(fake_carla):
    with carla_session(CarlaSimConfig()) as s:
        st = s.world.get_settings()
        assert st.synchronous_mode and st.fixed_delta_seconds == 0.05 and st.max_substep_delta_time == 0.01
        assert st.max_substeps == 10
        assert s.tm.synchronous is True and s.tm.seed == 0


def _spawn_sensor(s):
    sensor = s.world.spawn_actor(s.world.get_blueprint_library().find("sensor.other.imu"), None)
    sensor.listen(lambda d: None)
    s.actors.append(sensor)
    return sensor


def test_sync_mode_restored_on_exception(fake_carla):
    with pytest.raises(RuntimeError), carla_session(CarlaSimConfig()) as s:
        sensor = _spawn_sensor(s)
        raise RuntimeError("boom")
    w = fake_carla.last_world
    assert not w.get_settings().synchronous_mode and not fake_carla.last_tm.synchronous
    assert w.get_settings().fixed_delta_seconds is None
    assert sensor.stopped_before_destroy and sensor.destroyed


def test_sync_mode_restored_on_keyboard_interrupt(fake_carla):
    with pytest.raises(KeyboardInterrupt), carla_session(CarlaSimConfig()) as s:
        sensor = _spawn_sensor(s)
        raise KeyboardInterrupt
    assert not fake_carla.last_world.get_settings().synchronous_mode and sensor.destroyed


def test_clean_exit_restores_and_destroys(fake_carla):
    with carla_session(CarlaSimConfig()) as s:
        sensor = _spawn_sensor(s)
    assert not fake_carla.last_world.get_settings().synchronous_mode and sensor.destroyed


def test_restore_forces_async_even_if_server_was_left_synchronous(fake_carla):
    # a crashed earlier client left the server frozen in synchronous mode
    w = fake_carla.last_world
    stale = w.get_settings()
    stale.synchronous_mode, stale.fixed_delta_seconds = True, 0.05
    w.apply_settings(stale)
    with carla_session(CarlaSimConfig()):
        pass
    st = w.get_settings()
    assert not st.synchronous_mode and st.fixed_delta_seconds is None


def test_failure_inside_apply_sync_restores(fake_carla):
    client = fake_carla.Client()
    world, tm = client.get_world(), client.get_trafficmanager()
    tm.set_random_device_seed = lambda seed: (_ for _ in ()).throw(RuntimeError("tm down"))
    with pytest.raises(RuntimeError, match="tm down"):
        apply_sync(world, tm, CarlaSimConfig())
    assert not world.get_settings().synchronous_mode and not tm.synchronous


def test_restore_still_resets_world_if_tm_fails(fake_carla):
    client = fake_carla.Client()
    world, tm = client.get_world(), client.get_trafficmanager()
    original = apply_sync(world, tm, CarlaSimConfig())
    tm.set_synchronous_mode = lambda on: (_ for _ in ()).throw(RuntimeError("tm gone"))
    with pytest.raises(RuntimeError, match="tm gone"):
        restore(world, tm, original)
    assert not world.get_settings().synchronous_mode


def test_invalid_substep_config():
    with pytest.raises(ValueError, match="substep"):
        apply_sync(None, None, CarlaSimConfig(dt=0.2, substep_dt=0.01, max_substeps=10))


def test_town_not_reloaded_when_current(fake_carla):
    fake_carla.set_current_map("Carla/Maps/Town03_Opt")
    with carla_session(CarlaSimConfig(town="Town03_Opt")):
        pass
    assert fake_carla.load_world_calls == 0


def test_town_loaded_then_settings_applied_after(fake_carla):
    # load_world resets settings (ECC-02), so sync mode must be applied after it
    with carla_session(CarlaSimConfig(town="Town03_Opt")) as s:
        assert fake_carla.load_world_calls == 1
        assert s.world.get_settings().synchronous_mode
