import math
import sys

import numpy as np
import pytest

from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import FrictionZone, Scenario, ScriptedActor
from hybrid_av.sim.carla_backend.client import CarlaSimConfig, carla_session
from hybrid_av.sim.carla_backend.scenario_setup import (
    ActorDirector,
    FrictionController,
    FrictionNotApplied,
    build_route,
    route_lights,
)
from hybrid_av.types import ObjectClass


def test_route_from_fake_planner(fake_carla):
    fake_carla.planner_trace = fake_carla.straight_trace(
        n=101, spacing=1.0, lane_width=3.5
    )  # CARLA x = 0 ... 100 m
    r = build_route(fake_carla.last_or_new_world().get_map(), [0, 0, 0, 0], [100, 0, 0])
    assert r.length == pytest.approx(100.0, abs=0.5) and r.lane_width[0] == 3.5
    assert r.speed_limit[0] == pytest.approx(8.33, abs=0.01)


def test_route_converts_to_library_frame_and_uses_speed_cache(fake_carla):
    trace = fake_carla.straight_trace(n=11)
    for i, (wp, _) in enumerate(trace):
        wp.transform.location.y = -float(i)  # CARLA y right: heading drifts to the library's left
    fake_carla.planner_trace = trace
    r = build_route(
        fake_carla.last_or_new_world().get_map(), [0, 0, 0, 0], [10, -10, 0], cache={"speed_limit": 11.0}
    )
    assert r.y[-1] > 0 and r.speed_limit[0] == pytest.approx(11.0)


def test_route_rejects_lane_change(fake_carla):
    fake_carla.planner_trace = fake_carla.straight_trace(n=50, spacing=1.0, lane_width=3.5, lane_change_at=20)
    with pytest.raises(ValueError, match="lane change"):
        build_route(fake_carla.last_or_new_world().get_map(), [0, 0, 0, 0], [50, 0, 0])


def test_missing_agents_message(monkeypatch, fake_carla):
    monkeypatch.setitem(sys.modules, "agents.navigation.global_route_planner", None)
    with pytest.raises(ImportError, match="PYTHONPATH"):
        build_route(fake_carla.last_or_new_world().get_map(), [0, 0, 0, 0], [10, 0, 0])


def test_route_lights_match_stop_waypoints_to_route(fake_carla):
    import carla

    trace = fake_carla.straight_trace(n=101)
    wps = [wp for wp, _ in trace]
    world = fake_carla.last_or_new_world()
    on_route, off_route = (
        world.spawn_actor(carla.ActorBlueprint("traffic.traffic_light"), carla.Transform()) for _ in range(2)
    )
    on_route.stop_waypoints = [carla.Waypoint(carla.Transform(carla.Location(40.2, 0.3, 0.0)))]
    off_route.stop_waypoints = [carla.Waypoint(carla.Transform(carla.Location(40.0, 50.0, 0.0)))]
    (found,) = route_lights(world, wps)
    assert found.light is on_route and found.stop_s == pytest.approx(40.0, abs=1.0)


def test_friction_zone_applies_and_restores(fake_carla):
    v = fake_carla.spawn_vehicle(tire_friction=3.5)
    fc = FrictionController(v, [FrictionZone(50.0, 80.0, 0.35)], tire_scale=0.3)
    assert fc.update(40.0) is False
    assert fc.update(55.0) is True
    fc.verify()
    assert all(w.tire_friction == pytest.approx(3.5 * 0.3) for w in v.get_physics_control().wheels)
    assert fc.update(60.0) is False
    assert fc.update(90.0) is True
    fc.verify()
    assert all(w.tire_friction == pytest.approx(3.5) for w in v.get_physics_control().wheels)


def test_friction_scale_defaults_to_zone_mu_and_keeps_other_wheel_fields(fake_carla):
    v = fake_carla.spawn_vehicle(tire_friction=3.5)
    fc = FrictionController(v, [FrictionZone(0.0, 10.0, 0.35)])
    fc.update(5.0)
    fc.verify()
    (w, *_) = v.get_physics_control().wheels
    assert w.tire_friction == pytest.approx(3.5 * 0.35) and w.max_steer_angle == 70.0 and w.radius == 30.0


def test_friction_readback_mismatch_raises(fake_carla):
    v = fake_carla.spawn_vehicle(tire_friction=3.5)
    v.physics_apply_mode = "ignore"
    fc = FrictionController(v, [FrictionZone(0.0, 10.0, 0.35)])
    fc.update(5.0)
    with pytest.raises(FrictionNotApplied):
        fc.verify()


# --- ActorDirector --------------------------------------------------------------------------------------------


def _scenario(*actors, junction_s=()):
    return Scenario("t", 0, None, None, (), 1.0, tuple(actors), (), junction_s, 60.0, 0.0)


def _director(fake_carla, *actors, junction_s=()):
    s = carla_session(CarlaSimConfig())
    session = s.__enter__()
    director = ActorDirector(
        session.world,
        session.tm,
        _scenario(*actors, junction_s=junction_s),
        RouteBuilder.straight(300.0, 11.0),
        np.random.default_rng(0),
        actors=session.actors,
    )
    return session, director


def test_director_spawns_lead_parked_and_pedestrian(fake_carla):
    lead = ScriptedActor(1, ObjectClass.CAR, "lead", {"s0": 30.0, "v_cruise": 8.0, "brake_t": -1.0, "brake_to": 0.0, "brake_decel": 3.0})  # fmt: skip
    parked = ScriptedActor(
        2, ObjectClass.CAR, "parked", {"s": 50.0, "lateral": 3.0, "length": 4.5, "width": 1.9}
    )
    ped = ScriptedActor(3, ObjectClass.PEDESTRIAN, "pedestrian_crossing", {"s": 100.0, "trigger_dist": 30.0, "speed": 1.4, "side": 1.0})  # fmt: skip
    session, director = _director(fake_carla, lead, parked, ped)
    lead_v, parked_v, walker = director.spawn()
    assert lead_v.get_transform().location.x == pytest.approx(30.0) and lead_v.autopilot is True
    assert fake_carla.last_tm.desired_speed == pytest.approx(8.0 * 3.6)
    assert parked_v.get_transform().location.y == pytest.approx(-3.0)  # library +3 m left = CARLA y -3
    assert walker.type_id.startswith("walker.") and set(director.actors) == {lead_v, parked_v, walker}
    assert director.actors is session.actors


def test_director_pedestrian_walks_when_triggered(fake_carla):
    ped = ScriptedActor(3, ObjectClass.PEDESTRIAN, "pedestrian_crossing", {"s": 100.0, "trigger_dist": 30.0, "speed": 1.4, "side": 1.0})  # fmt: skip
    _, director = _director(fake_carla, ped)
    (walker,) = director.spawn()
    director.tick(0.05, 10.0)
    assert not hasattr(walker, "control") or walker.control.speed == 0.0
    director.tick(0.10, 80.0)  # within trigger_dist
    assert walker.control.speed == pytest.approx(1.4)
    d = walker.control.direction  # side +1 walks towards library -y = CARLA +y
    assert d.x == pytest.approx(0.0, abs=1e-9) and d.y == pytest.approx(1.0)


def test_director_lead_braking_via_traffic_manager(fake_carla):
    lead = ScriptedActor(1, ObjectClass.CAR, "lead", {"s0": 30.0, "v_cruise": 8.0, "brake_t": 2.0, "brake_to": 2.0, "brake_decel": 3.0, "brake_hold": 4.0})  # fmt: skip
    _, director = _director(fake_carla, lead)
    director.spawn()
    director.tick(1.0, 0.0)
    assert fake_carla.last_tm.desired_speed == pytest.approx(8.0 * 3.6)
    director.tick(2.1, 0.0)
    assert fake_carla.last_tm.desired_speed == pytest.approx(2.0 * 3.6)
    director.tick(8.5, 0.0)  # hold (reached at t = 4 s, 4 s long) is over: back to cruise
    assert fake_carla.last_tm.desired_speed == pytest.approx(8.0 * 3.6)


def test_director_cross_stream_is_driven_every_tick_and_destroyed_after_passing(fake_carla):
    stream = ScriptedActor(1, ObjectClass.CAR, "cross_stream", {"junction_index": 0, "direction": 1.0, "speed": 10.0, "gap_mean": 3.0, "gap_min": 2.5, "path_offset": 3.0})  # fmt: skip
    session, director = _director(fake_carla, stream, junction_s=((100.0, 110.0),))
    director.spawn()
    seen: set[int] = set()
    for k in range(1, 400):
        director.tick(0.05 * k, 0.0)
        for v in session.actors:
            seen.add(v.id)
            vel = v.target_velocity  # re-applied every tick, no autopilot
            assert math.hypot(vel.x, vel.y) == pytest.approx(10.0)
    assert seen
    assert len(session.actors) < len(seen)  # some passed the junction and were destroyed
    assert all(not hasattr(v, "autopilot") for v in session.actors)


def _xs_trace(fake_carla, xs):
    trace = fake_carla.straight_trace(n=len(xs))
    for (wp, _), x in zip(trace, xs, strict=True):
        wp.transform.location.x = float(x)
    return trace


def test_route_drops_planner_waypoints_that_step_backwards(fake_carla):
    # IC-17: the planner's first waypoint (x=5) is followed by ones behind it (3, 4) before the route moves on;
    # kept as is, the polyline reverses and the heading jumps by pi
    fake_carla.planner_trace = _xs_trace(fake_carla, [5, 3, 4, 5, 6, 7, 8, 9, 10])
    r = build_route(fake_carla.last_or_new_world().get_map(), [5, 0, 0, 0], [10, 0, 0])
    assert np.max(np.abs(np.diff(r.psi))) < 0.3
    assert r.length == pytest.approx(5.0, abs=0.5) and r.x[0] == pytest.approx(5.0)


def test_route_rejects_a_start_far_from_the_first_waypoint(fake_carla):
    # IC-37: the planner snapped to a road 7 m from the spawn pose, so the ego never saw the route
    fake_carla.planner_trace = _xs_trace(fake_carla, [7, 8, 9, 10])
    with pytest.raises(ValueError, match="start"):
        build_route(fake_carla.last_or_new_world().get_map(), [0, 0, 0, 0], [10, 0, 0])
