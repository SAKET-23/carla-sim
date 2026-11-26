from pathlib import Path

import pytest

from hybrid_av.sim.base import FrictionZone, Sim, TrafficControl
from hybrid_av.sim.carla_backend.client import CarlaSimConfig
from hybrid_av.sim.carla_backend.scenario_setup import FrictionNotApplied
from hybrid_av.sim.carla_backend.sim import CarlaSim
from hybrid_av.types import ActuatorCommand, TrafficLightState
from tests.sim.carla_backend.scenarios import straight_nominal

RIG = Path(__file__).resolve().parents[3] / "configs/sensors/carla_rig.yaml"
IDLE = ActuatorCommand(0.0, 0.0, 0.0)


_open: list[CarlaSim] = []


@pytest.fixture(autouse=True)
def _close_sims(fake_carla):
    yield
    while _open:
        _open.pop().close()


def _sim(**kw):
    sim = CarlaSim(CarlaSimConfig(sensors=None, **kw))
    _open.append(sim)
    return sim


def _move(fake_carla, x, y_carla=0.0):
    fake_carla.ego.transform.location.x, fake_carla.ego.transform.location.y = x, y_carla


def test_implements_sim_protocol(fake_carla):
    assert isinstance(_sim(), Sim) and _sim().dt == 0.05


def test_reset_spawns_and_settles(fake_carla):
    sim = _sim()
    obs = sim.reset(straight_nominal())
    assert fake_carla.last_world.frame >= 10 and obs.ego.vx == pytest.approx(0.0) and obs.ego.t == 0.0
    assert sim.route().length > 50
    assert obs.source == "ground_truth" and obs.friction_true == 1.0
    assert obs.context.speed_limit == pytest.approx(8.33, abs=0.01) and obs.context.lane_width == 3.5


def test_reset_reads_steer_scale_from_physics_control(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal())
    assert sim.max_steer == pytest.approx(70.0 * 3.141592653589793 / 180)  # ECC-05: wheels[0].max_steer_angle


def test_step_applies_control_with_steer_scale(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal())
    sim.step(ActuatorCommand(steer=-0.5, throttle=0.3, brake=0.0))
    c = fake_carla.ego.last_control
    assert (c.steer, c.throttle, c.brake) == (-0.5, 0.3, 0.0)


def test_step_reports_sim_time_and_route_progress(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal())
    _move(fake_carla, 10.0)
    obs, info = sim.step(IDLE)
    assert info.sim_time == pytest.approx(0.05) and obs.ego.t == pytest.approx(0.05)
    assert info.route_s == pytest.approx(10.0, abs=0.5) and not info.route_completed
    _move(fake_carla, 5.0)  # going back never lowers completion
    assert sim.step(IDLE)[1].route_s == pytest.approx(10.0, abs=0.5)
    for x in (30.0, 50.0, 70.0, 90.0, 99.5):  # projection searches [s - 10, s + 30], so no teleporting
        _move(fake_carla, x)
        info = sim.step(IDLE)[1]
    assert info.route_completed


def test_collision_classification_and_debounce(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal())
    fake_carla.collide(other_type="vehicle.audi.tt", other_id=42)
    _, info = sim.step(IDLE)
    assert info.collision == "vehicle"
    fake_carla.collide(other_type="vehicle.audi.tt", other_id=42)
    _, info = sim.step(IDLE)
    assert info.collision == "none"  # same actor within 1 s
    fake_carla.collide(other_type="", other_id=0)
    _, info = sim.step(IDLE)
    assert info.collision == "static"
    fake_carla.collide(other_type="walker.pedestrian.0001", other_id=7)
    assert sim.step(IDLE)[1].collision == "pedestrian"


def test_off_road_needs_both_a_non_driving_lane_and_lateral_distance(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal())
    _move(fake_carla, 20.0, y_carla=-8.0)  # 8 m left of the route, beyond 1.5 lanes
    assert sim.step(IDLE)[1].off_road is False  # still on a Driving lane (adjacent lane stays legal)
    fake_carla.last_world.get_map().drivable = False
    assert sim.step(IDLE)[1].off_road is True
    _move(fake_carla, 20.0, y_carla=-1.0)
    assert sim.step(IDLE)[1].off_road is False  # off the lane but within the lateral threshold


def test_route_deviation_and_timeout(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal(time_limit=0.07))
    _move(fake_carla, 20.0, y_carla=40.0)
    _, info = sim.step(IDLE)
    assert info.route_deviation and not info.timeout
    assert sim.step(IDLE)[1].timeout


def test_red_light_violation_uses_precomputed_lights(fake_carla):
    import carla

    light = fake_carla.last_world.spawn_actor(
        carla.ActorBlueprint("traffic.traffic_light"), carla.Transform()
    )
    light.stop_waypoints = [carla.Waypoint(carla.Transform(carla.Location(30.0, 0.0, 0.0)))]
    light.state = carla.TrafficLightState.Red
    sim = _sim()
    obs = sim.reset(straight_nominal(controls=(TrafficControl("light", 30.0, phase_offset=-14.0),)))
    assert (
        obs.context.light_state == TrafficLightState.RED
        and obs.context.dist_to_stop_line == pytest.approx(30.0 - 2.4, abs=1.0)
    )
    _move(fake_carla, 20.0)
    assert not sim.step(IDLE)[1].red_light_violation
    _move(fake_carla, 40.0)  # the front bumper crosses the line while red
    assert sim.step(IDLE)[1].red_light_violation
    assert (light.green, light.yellow, light.red) == (
        10.0,
        3.0,
        10.0,
    )  # timings come from the TrafficControl cycle
    light.state = carla.TrafficLightState.Green
    _move(fake_carla, 45.0)
    assert not sim.step(IDLE)[1].red_light_violation


def test_stop_sign_rule_is_litesims(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal(controls=(TrafficControl("stop_sign", 30.0),)))
    _move(fake_carla, 20.0)
    sim.step(IDLE)
    _move(
        fake_carla, 40.0
    )  # crossed without stopping (speed 0 in the fake is not near the line, so unserved)
    assert sim.step(IDLE)[1].stop_sign_violation


def test_context_junction_distance_from_route_waypoints(fake_carla):
    trace = fake_carla.straight_trace(n=101)
    for wp, _ in trace[60:71]:
        wp.is_junction = True
    fake_carla.planner_trace = trace
    obs = _sim().reset(straight_nominal())
    assert obs.context.dist_to_junction == pytest.approx(60.0 - 2.4, abs=1.0)


def test_friction_zone_applied_and_reported(fake_carla):
    sim = _sim(tire_friction_scale=0.3)
    sim.reset(straight_nominal(friction_zones=(FrictionZone(20.0, 40.0, 0.35),), base_mu=1.0))
    _move(fake_carla, 25.0)
    obs, _ = sim.step(IDLE)  # entering the zone: applied after this step
    assert obs.friction_true == 0.35
    obs, _ = sim.step(IDLE)  # next tick: read back and checked
    assert all(
        w.tire_friction == pytest.approx(3.5 * 0.3) for w in fake_carla.ego.get_physics_control().wheels
    )
    assert obs.context.surface_roughness == pytest.approx(0.8)


def test_friction_not_applied_raises_and_restores_server(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal(friction_zones=(FrictionZone(20.0, 40.0, 0.35),)))
    fake_carla.ego.physics_apply_mode = "ignore"
    _move(fake_carla, 25.0)
    sim.step(IDLE)
    with pytest.raises(FrictionNotApplied):
        sim.step(IDLE)
    assert not fake_carla.last_world.get_settings().synchronous_mode


def test_close_restores_async(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal())
    sim.close()
    assert not fake_carla.last_world.get_settings().synchronous_mode and not fake_carla.last_tm.synchronous
    sim.close()  # idempotent


def test_failed_reset_restores_async(fake_carla):
    sim = _sim()
    sc = straight_nominal(carla_route={"town": "Town03", "start": [0, 0, 0, 0], "goal": [10, 0, 0]})
    fake_carla.planner_trace = fake_carla.straight_trace(n=10, lane_change_at=3)
    with pytest.raises(ValueError, match="lane change"):
        sim.reset(sc)
    assert not fake_carla.last_world.get_settings().synchronous_mode


def test_second_reset_destroys_previous_actors_without_reloading(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal())
    first_ego, loads = fake_carla.ego, fake_carla.load_world_calls
    sim.reset(straight_nominal())
    assert first_ego.destroyed and fake_carla.load_world_calls == loads
    assert fake_carla.last_world.get_settings().synchronous_mode


def test_sensor_data_exposed_per_frame(fake_carla):
    sim = CarlaSim(CarlaSimConfig(sensors=RIG))
    _open.append(sim)
    sim.reset(straight_nominal())
    sim.step(IDLE)
    assert set(sim.last_sensor_data) >= {"camera_front", "lidar", "radar_front", "imu"}
    assert sim.last_sensor_data["lidar"].frame == sim.last_frame
    assert [d.frame for d in sim.last_sensor_data["imu"]] == [sim.last_frame]


def test_no_sensor_rig_has_empty_sensor_data_but_still_collision_sensor(fake_carla):
    sim = _sim()
    sim.reset(straight_nominal())
    sim.step(IDLE)
    assert sim.last_sensor_data == {}
    assert any(a.type_id == "sensor.other.collision" for a in fake_carla.last_world.actors.values())


def test_reset_sets_ego_start_speed_along_heading(fake_carla):
    sc = straight_nominal(ego_start_speed=5.0)
    sc.carla_route["start"] = [0.0, 0.0, 0.0, 90.0]  # CARLA yaw 90 deg: heading along world +y
    sim = CarlaSim(CarlaSimConfig(sensors=None))
    sim.reset(sc)
    v = sim.ego.target_velocity
    assert (v.x, v.y, v.z) == pytest.approx((0.0, 5.0, 0.0), abs=1e-9)
    sim.close()
