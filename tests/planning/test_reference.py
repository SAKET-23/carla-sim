import numpy as np

from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig, ReferenceGenerator
from hybrid_av.planning.route import RouteBuilder
from hybrid_av.types import EgoState, PolicyAction

G = ReferenceGenerator(ReferenceConfig(), MPCConfig())
A1 = PolicyAction(alpha_v=1.0, delta_y=0.0, brake=0.0)


def _ego(x=0.0, y=0.0, psi=0.0, vx=10.0):
    return EgoState(t=0, x=x, y=y, psi=psi, vx=vx, vy=0, r=0)


def test_rule1_alpha_scales_limit():
    ref = G.generate(
        RouteBuilder.straight(300, 13.89), _ego(vx=16.668), 0.0, PolicyAction(1.2, 0.0, 0.0), 1.0
    )
    assert len(ref.v) == 21 and np.all(ref.v <= 1.2 * 13.89 + 1e-9)
    assert ref.v[-1] > 16.0  # the cap is actually raised above the 13.89 limit
    # alpha 0.8: from 12 m/s the 2 m/s² ramp reaches the 11.112 cap in 0.44 s, inside the 1 s horizon
    low = G.generate(RouteBuilder.straight(300, 13.89), _ego(vx=12.0), 0.0, PolicyAction(0.8, 0.0, 0.0), 1.0)
    assert low.v[-1] < 0.8 * 13.89 + 1e-6 and low.v[-1] > 0.8 * 13.89 - 1e-6


def test_rule1_curvature_cap_scales_with_mu():
    arc = RouteBuilder.arc(30.0, np.pi, 30.0)
    v_dry, v_wet = np.sqrt(4.5 * 30), np.sqrt(0.85 * 0.35 * 9.81 * 30)  # 11.619, 9.357; ref.v[0] = ego.vx
    dry = G.generate(arc, _ego(vx=v_dry), 1.0, A1, mu_est=1.0)
    wet = G.generate(arc, _ego(vx=v_wet), 1.0, A1, mu_est=0.35)
    assert np.all(dry.v <= np.sqrt(4.5 * 30) + 1e-6)  # a_lat_max binds: 11.619
    assert np.all(wet.v <= np.sqrt(0.85 * 0.35 * 9.81 * 30) + 1e-6)  # friction binds: 9.357


def test_rule1_lookahead_brakes_before_curve():  # mpc-formulation.md §7 rule 1 window
    route = RouteBuilder.concat(RouteBuilder.straight(100, 13.89), RouteBuilder.arc(30.0, np.pi / 2, 13.89))
    ref = G.generate(route, _ego(x=90.0, vx=13.89), 90.0, A1, 1.0)
    assert ref.s[5] < 100.0 and ref.v[5] < 13.89 - 0.4 and ref.v[-1] < 12.9  # slows on the straight already


def test_rule2_comfort_ramp():
    ref = G.generate(RouteBuilder.straight(300, 13.89), _ego(vx=0.0), 0.0, A1, 1.0)
    np.testing.assert_allclose(ref.v, 0.1 * np.arange(21), atol=1e-9)  # 2.0 m/s² × 0.05 s per step


def test_rule3_stop_point():
    ref = G.generate(RouteBuilder.straight(300, 13.89), _ego(vx=10.0), 0.0, A1, 1.0, stop_s=15.0)
    cap = np.sqrt(4.0 * np.maximum(13.0 - ref.s, 0.0))  # 2·a_lon_comfort·(stop_s − stop_buffer − s)
    assert np.all(ref.v <= cap + 1e-9) and ref.v[0] < 9.0 and np.all(ref.s <= 13.0 + 1e-9)


def test_actuator_lag_brakes_earlier_same_stop_point():
    route, kw = RouteBuilder.straight(300, 13.89), {"stop_s": 15.0}
    base = G.generate(route, _ego(vx=10.0), 0.0, A1, 1.0, **kw)
    same = ReferenceGenerator(ReferenceConfig(actuator_lag=0.0), MPCConfig()).generate(
        route, _ego(vx=10.0), 0.0, A1, 1.0, **kw
    )
    lag = ReferenceGenerator(ReferenceConfig(actuator_lag=0.5), MPCConfig()).generate(
        route, _ego(vx=10.0), 0.0, A1, 1.0, **kw
    )
    np.testing.assert_array_equal(base.v, same.v)
    assert lag.v[0] < base.v[0] and np.all(lag.v <= base.v + 1e-9)
    # the stop point itself is unchanged: at 13 m (stop_s - stop_buffer) both hold v = 0
    for cfg in (ReferenceConfig(), ReferenceConfig(actuator_lag=0.5)):
        at_stop = ReferenceGenerator(cfg, MPCConfig()).generate(route, _ego(vx=0.0), 13.0, A1, 1.0, **kw)
        assert np.all(at_stop.v == 0.0)


def test_rule3_brake_action():
    ref = G.generate(RouteBuilder.straight(300, 13.89), _ego(vx=10.0), 0.0, PolicyAction(1.0, 0.0, 0.8), 1.0)
    assert np.all(np.diff(ref.v) <= 1e-9) and ref.v[-1] < ref.v[0] - 1.0


def test_stop_behind_ego_ignored():
    ref = G.generate(RouteBuilder.straight(300, 13.89), _ego(vx=10.0), 50.0, A1, 1.0, stop_s=45.0)
    assert ref.v[-1] >= 10.0 - 1e-9


def test_rule4_lateral_ramp_and_clip():
    r = RouteBuilder.straight(300, 13.89)
    ref = G.generate(r, _ego(), 0.0, PolicyAction(1.0, 3.5, 0.0), 1.0)
    lat = np.array([r.project(x, y).lateral for x, y in zip(ref.x, ref.y)])
    np.testing.assert_allclose(lat, 0.05 * np.arange(21), atol=1e-6)  # 1.0 m/s × 0.05 s per step
    ref2 = G.generate(r, _ego(y=5.2), 0.0, PolicyAction(1.0, 3.5, 0.0), 1.0)
    assert np.all(np.array([r.project(x, y).lateral for x, y in zip(ref2.x, ref2.y)]) <= 5.25 + 1e-9)
    ref3 = G.generate(
        r, _ego(y=5.2), 0.0, PolicyAction(1.0, 10.0, 0.0), 1.0
    )  # clip binds at lane_width * 1.5
    lat3 = np.array([r.project(x, y).lateral for x, y in zip(ref3.x, ref3.y)])
    assert np.all(lat3 <= 5.25 + 1e-9) and abs(lat3[-1] - 5.25) < 1e-6


def test_negative_mu_finite():
    ref = G.generate(RouteBuilder.arc(30.0, np.pi, 30.0), _ego(), 1.0, A1, mu_est=-0.1)
    for a in (ref.x, ref.y, ref.psi, ref.v, ref.kappa, ref.s):
        assert np.all(np.isfinite(a))


def test_brake_while_stopped_holds_zero():
    ref = G.generate(RouteBuilder.straight(300, 13.89), _ego(vx=0.0), 0.0, PolicyAction(1.0, 0.0, 0.9), 1.0)
    assert np.all(ref.v == 0.0)


def test_rule5_heading_continuous_on_lap3():
    loop = RouteBuilder.circle_loop(30.0, 3.0, 10.0)
    s0 = 2 * 2 * np.pi * 30 + 3.0
    d = loop.sample(np.array([s0]))
    ego = _ego(x=d["x"][0], y=d["y"][0], psi=4 * np.pi + 0.1)
    ref = G.generate(loop, ego, s0, A1, 1.0)
    assert abs(ref.psi[0] - ego.psi) <= np.pi and np.all(np.abs(np.diff(ref.psi)) < 0.1)
    wrapped = G.generate(
        loop, _ego(x=d["x"][0], y=d["y"][0], psi=0.1), s0, A1, 1.0
    )  # same pose, heading wrapped
    assert abs(wrapped.psi[0] - 0.1) < 1e-3 and np.allclose(np.diff(wrapped.psi), np.diff(ref.psi))
