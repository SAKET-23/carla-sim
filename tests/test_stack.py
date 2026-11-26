from dataclasses import replace

import numpy as np

from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.planning.route import RouteBuilder
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import (
    Control,
    EgoState,
    ObjectClass,
    ObjectState,
    Observation,
    PolicyAction,
    TrafficContext,
)
from hybrid_av.types import TrafficLightState as L
from hybrid_av.vehicle.actuation import LongitudinalMap, control_to_actuator
from hybrid_av.vehicle.params import VehicleParams


def _obs_on(route, s, vx, friction_true=1.0, objects=()):
    d = route.sample(np.array([s]))
    ego = EgoState(0.0, float(d["x"][0]), float(d["y"][0]), float(d["psi"][0]), vx, 0.0, 0.0)
    ctx = TrafficContext(L.NONE, np.inf, np.inf, np.inf, float(d["speed_limit"][0]), 3.5, 0.0, 0.0)
    return Observation(ego, tuple(objects), ctx, friction_true, "ground_truth")


def _stack(route):
    st = AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig())
    st.reset(route)
    return st


def _spy(st, monkeypatch):
    seen, orig = {}, st.mpc.solve

    def spy(*a, **k):
        seen.update(k)
        return orig(*a, **k)

    monkeypatch.setattr(st.mpc, "solve", spy)
    return seen


def test_uses_estimate_not_truth():
    r = RouteBuilder.arc(30.0, np.pi, 30.0)
    out = _stack(r).act(_obs_on(r, s=5.0, vx=9.0, friction_true=0.1), PolicyAction(1.0, 0.0, 0.0))
    assert out.features.mu_est == 1.0 and np.max(out.reference.v) > 9.0  # truth would cap at 5.0 m/s


def test_brake_action_sets_request(monkeypatch):
    r = RouteBuilder.straight(300, 13.89)
    st = _stack(r)
    seen = _spy(st, monkeypatch)
    st.act(_obs_on(r, s=10.0, vx=10.0), PolicyAction(1.0, 0.0, 0.9))
    assert seen["brake_request"] is True


def test_obstacle_selection_capped(monkeypatch):
    r = RouteBuilder.straight(300, 13.89)
    st = _stack(r)
    seen = _spy(st, monkeypatch)
    parked = [
        ObjectState(i, ObjectClass.CAR, 10.0 + 10.0 * i, 0.0, 0.0, 0.0, 4.5, 1.9, 1.5, 0.0, 0.0)
        for i in range(1, 7)
    ]
    far = ObjectState(9, ObjectClass.CAR, 15.0, 40.0, 0.0, 0.0, 4.5, 1.9, 1.5, 0.0, 0.0)  # 40 m off-route
    st.act(_obs_on(r, s=0.0, vx=5.0, objects=parked + [far]), PolicyAction(1.0, 0.0, 0.0))
    assert [o.id for o in seen["obstacles"]] == [1, 2, 3, 4]


def test_command_from_actuation():
    r = RouteBuilder.straight(300, 13.89)
    out = _stack(r).act(_obs_on(r, s=10.0, vx=10.0), PolicyAction(1.0, 0.0, 0.0))
    assert out.command == control_to_actuator(out.control, 10.0, VehicleParams(), LongitudinalMap())


def test_red_light_stop_keeps_bumper_behind_line():
    from hybrid_av.sim.base import Scenario, TrafficControl
    from hybrid_av.sim.litesim.sim import LiteSim

    r, line = RouteBuilder.straight(300, 13.89), 40.0
    light = TrafficControl("light", line, cycle=(0.0, 0.0, 1000.0))
    sc = Scenario("t", 0, r, None, (), 1.0, (), (light,), (), 60.0, 8.0)
    sim, st, p = LiteSim(VehicleParams(), LongitudinalMap()), _stack(r), VehicleParams()
    obs = sim.reset(sc)
    bumper = []
    for _ in range(int(sc.time_limit / sim.dt)):
        out = st.act(obs, PolicyAction(1.0, 0.0, 0.0))
        obs, _ = sim.step(out.command)
        bumper.append(obs.ego.x + p.a + 0.5)  # straight route along x
    assert max(bumper) <= line and obs.ego.vx < 0.1
    assert 0.5 <= line - bumper[-1] <= 3.0, line - bumper[-1]


def test_standstill_hold_zeroes_throttle(monkeypatch):
    r = RouteBuilder.straight(300, 13.89)
    st = _stack(r)
    creep = Control(0.0, 2.0)  # an MPC throttle burst while stopped
    orig = st.mpc.solve
    monkeypatch.setattr(st.mpc, "solve", lambda *a, **k: replace(orig(*a, **k), control=creep))
    stopped = st.act(_obs_on(r, s=10.0, vx=0.2), PolicyAction(1.0, 0.0, 1.0)).command
    assert stopped.throttle == 0.0 and stopped.brake >= 0.5
    moving = st.act(_obs_on(r, s=10.0, vx=5.0), PolicyAction(1.0, 0.0, 1.0)).command
    assert moving.throttle > 0.0  # above the hold speed the MPC output is untouched
    no_req = st.act(_obs_on(r, s=10.0, vx=0.2), PolicyAction(1.0, 0.0, 0.0)).command
    assert no_req.throttle > 0.0  # no stop request, no hold
