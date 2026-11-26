import numpy as np

from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import Scenario, ScriptedActor
from hybrid_av.sim.litesim.actors import ActorManager
from hybrid_av.types import EgoState, ObjectClass

R = RouteBuilder.straight(400, 13.89)
EGO = EgoState(0, 0, 0, 0, 10, 0, 0)


def _mgr(actors, junction_s=(), seed=0):
    sc = Scenario("t", seed, R, None, (), 1.0, actors, (), junction_s, 600, 10)
    return ActorManager(sc, R, np.random.default_rng(seed))


def test_lead_brake_event():
    m = _mgr(
        (
            ScriptedActor(
                1,
                ObjectClass.CAR,
                "lead",
                {"s0": 30, "v_cruise": 10, "brake_t": 2.0, "brake_to": 0.0, "brake_decel": 4.0},
            ),
        )
    )
    speeds = [np.hypot(o.vx, o.vy) for o in (m.step(k * 0.05, 0.05, EGO, 0.0)[0] for k in range(120))]
    assert abs(speeds[30] - 10) < 1e-6 and speeds[-1] == 0.0 and min(np.diff(speeds)) >= -4.0 * 0.05 - 1e-9


def test_cross_stream_gap_statistics():
    m = _mgr(
        (
            ScriptedActor(
                2,
                ObjectClass.CAR,
                "cross_stream",
                {
                    "junction_index": 0,
                    "direction": 1,
                    "speed": 10,
                    "gap_mean": 3.5,
                    "gap_min": 2.5,
                    "path_offset": 3.0,
                },
            ),
        ),
        junction_s=((100.0, 110.0),),
    )
    gaps = m.sample_gaps(0, 4000)  # exposed for testing: next 4000 headways of stream 0
    assert gaps.min() >= 2.5 and abs(gaps.mean() - 3.5) < 0.1


def test_pedestrian_triggered_and_parked_static():
    m = _mgr(
        (
            ScriptedActor(
                3,
                ObjectClass.PEDESTRIAN,
                "pedestrian_crossing",
                {"s": 60, "trigger_dist": 30, "speed": 1.2, "side": 1},
            ),
            ScriptedActor(
                4, ObjectClass.CAR, "parked", {"s": 80, "lateral": -3.0, "length": 4.5, "width": 1.9}
            ),
        )
    )
    first = m.step(0.0, 0.05, EGO, 0.0)
    assert first[0].vy == 0.0  # ego 60 m away: pedestrian waits
    walking = m.step(0.05, 0.05, EgoState(0, 35, 0, 0, 10, 0, 0), 35.0)
    assert abs(np.hypot(walking[0].vx, walking[0].vy) - 1.2) < 1e-9 and walking[1].vx == 0.0 == walking[1].vy


def test_seeded():
    cs = (
        ScriptedActor(
            2,
            ObjectClass.CAR,
            "cross_stream",
            {
                "junction_index": 0,
                "direction": 0,
                "speed": 10,
                "gap_mean": 3.5,
                "gap_min": 2.5,
                "path_offset": 3.0,
            },
        ),
    )
    a, b = _mgr(cs, ((100.0, 110.0),), seed=5), _mgr(cs, ((100.0, 110.0),), seed=5)
    ra = [a.step(k * 0.05, 0.05, EGO, 0.0) for k in range(200)]
    assert ra == [b.step(k * 0.05, 0.05, EGO, 0.0) for k in range(200)] and any(len(x) > 0 for x in ra)


def test_lead_keeps_moving_past_route_end():
    m = _mgr(
        (
            ScriptedActor(
                1,
                ObjectClass.CAR,
                "lead",
                {"s0": 390, "v_cruise": 10, "brake_t": -1.0, "brake_to": 0.0, "brake_decel": 1.0},
            ),
        )
    )
    xs = [m.step(k * 0.05, 0.05, EGO, 0.0)[0].x for k in range(60)]
    assert xs[-1] > R.length + 10 and np.all(np.diff(xs) > 0)  # drives off the end instead of parking on it


def test_lead_recovers_after_brake_hold():
    m = _mgr(
        (
            ScriptedActor(
                1,
                ObjectClass.CAR,
                "lead",
                {
                    "s0": 30,
                    "v_cruise": 10,
                    "brake_t": 1.0,
                    "brake_to": 2.0,
                    "brake_decel": 4.0,
                    "brake_hold": 2.0,
                },
            ),
        )
    )
    v = [np.hypot(o.vx, o.vy) for o in (m.step(k * 0.05, 0.05, EGO, 0.0)[0] for k in range(400))]
    assert min(v) == 2.0 and abs(v[-1] - 10) < 1e-9
    assert abs(v[int(4.0 / 0.05)] - 2.0) < 1e-9  # still holding at t = 4 s (reached at 3 s, hold ends 5 s)
