from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import Scenario, ScriptedActor, TrafficControl
from hybrid_av.sim.litesim.rules import RuleChecker, light_state
from hybrid_av.types import EgoState, ObjectClass, ObjectState
from hybrid_av.types import TrafficLightState as L
from hybrid_av.vehicle.params import VehicleParams

R = RouteBuilder.straight(200, 13.89)


def _rc(controls=(), tl=600.0):
    return RuleChecker(Scenario("t", 0, R, None, (), 1.0, (), controls, (), tl, 10), R, VehicleParams())


def _e(x, y=0.0, vx=10.0, t=0.0):
    return EgoState(t, x, y, 0, vx, 0, 0)


def _o(cls, x, y=0.0):
    ped = cls == ObjectClass.PEDESTRIAN
    return ObjectState(1, cls, x, y, 0, 0, 0.5 if ped else 4.5, 0.5 if ped else 1.9, 1.5, 0, 0)


def test_light_cycle():
    c = TrafficControl("light", 50.0, (10.0, 3.0, 10.0), phase_offset=0.0)
    got = [light_state(c, t) for t in (0, 9.9, 10.1, 13.1, 22.9, 23.1)]
    assert got == [L.GREEN, L.GREEN, L.YELLOW, L.RED, L.RED, L.GREEN]


def test_collision_classes():
    assert _rc().update(0, _e(50), 50, 0, [_o(ObjectClass.PEDESTRIAN, 51.0)]).collision == "pedestrian"
    assert _rc().update(0, _e(50), 50, 0, [_o(ObjectClass.CAR, 53.0)]).collision == "vehicle"
    assert _rc().update(0, _e(50), 50, 0, [_o(ObjectClass.CAR, 60.0)]).collision == "none"


def test_collision_with_parked_is_static():
    sc = Scenario(
        "t", 0, R, None, (), 1.0, (ScriptedActor(1, ObjectClass.CAR, "parked", {}),), (), (), 600.0, 10
    )
    rc = RuleChecker(sc, R, VehicleParams())
    assert rc.update(0, _e(50), 50, 0, [_o(ObjectClass.CAR, 60.0)]).collision == "none"
    assert rc.update(0, _e(57), 57, 0, [_o(ObjectClass.CAR, 60.0)]).collision == "static"


def test_red_light_crossing():
    red = TrafficControl("light", 50.0, (10.0, 3.0, 10.0), phase_offset=-13.5)  # red at t = 0
    rc = _rc((red,))
    rc.update(0, _e(47.0), 47.0, 0, [])
    assert rc.update(0.05, _e(49.0), 49.0, 0, []).red_light_violation  # front bumper (49 + 1.7) crosses 50


def test_stop_sign_served():
    rc = _rc((TrafficControl("stop_sign", 50.0),))
    rc.update(0, _e(47.5, vx=0.3), 47.5, 0, [])  # front bumper 49.2: within 3 m, below 0.5 m/s
    assert rc.served_stop_signs() == {0}
    assert not rc.update(1, _e(49.0), 49.0, 0, []).stop_sign_violation
    rc2 = _rc((TrafficControl("stop_sign", 50.0),))
    rc2.update(0, _e(47.0), 47.0, 0, [])
    assert rc2.update(0.05, _e(49.0), 49.0, 0, []).stop_sign_violation


def test_off_road_deviation_blocked_timeout_completion():
    assert _rc().update(0, _e(50, y=5.3), 50, 5.3, []).off_road
    assert _rc().update(0, _e(50, y=31.0), 50, 31.0, []).route_deviation
    rc = _rc()
    info = None
    for k in range(3700):  # 185 s standing still
        info = rc.update(k * 0.05, _e(100, vx=0.0, t=k * 0.05), 100, 0, [])
    assert info.blocked
    assert _rc(tl=10.0).update(10.0, _e(50), 50, 0, []).timeout
    rc = _rc()
    rc.update(0, _e(199.9), 199.9, 0, [])
    assert rc.update(0.05, _e(150), 150, 0, []).route_s == 199.9  # monotone completion


# --- pure checks shared with CarlaSim -------------------------------------------------------------------------


def test_shared_checks():
    from hybrid_av.sim.litesim.rules import (
        BlockedMonitor,
        RouteProgress,
        StopSignMonitor,
        crossed,
        deviation,
        off_road_lateral,
        route_completed,
        timeout,
    )

    assert crossed(9.0, 10.5, 10.0) and not crossed(None, 10.5, 10.0) and not crossed(10.0, 11.0, 10.0)
    p = RouteProgress()
    assert (p.update(5.0), p.update(3.0), p.update(7.0)) == (5.0, 5.0, 7.0)
    b = BlockedMonitor()
    assert not b.update(0.0, 0.0, False) and b.update(180.0, 0.0, False)
    assert not b.update(181.0, 0.0, True) and not b.update(
        360.0, 0.0, True
    )  # waiting at a red light never blocks
    assert deviation(31.0) and not deviation(29.0) and timeout(5.0, 5.0) and not timeout(4.9, 5.0)
    assert off_road_lateral(5.3, 3.5) and not off_road_lateral(5.2, 3.5) and route_completed(99.0, 100.0)
    s = StopSignMonitor((TrafficControl("stop_sign", 10.0),))
    assert not s.update(9.0, 8.0, 0.0) and s.served == {0}  # served within 3 m below 0.5 m/s
    assert not s.update(10.5, 9.0, 3.0)  # crossing after being served is fine
    assert StopSignMonitor((TrafficControl("stop_sign", 10.0),)).update(10.5, 9.0, 3.0)  # unserved crossing
