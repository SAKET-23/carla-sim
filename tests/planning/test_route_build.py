import numpy as np

from hybrid_av.planning.route import RouteBuilder


def test_straight():
    r = RouteBuilder.straight(100.0, 13.89)
    assert abs(r.length - 100.0) < 1e-6 and np.all(r.kappa == 0) and np.all(np.diff(r.s) <= 1.0)
    assert np.all(r.speed_limit == 13.89) and np.all(r.lane_width == 3.5)


def test_arc_left_and_right():
    r = RouteBuilder.arc(30.0, np.pi / 2, 13.89)
    assert abs(r.length - 15 * np.pi) / (15 * np.pi) < 1e-3
    np.testing.assert_allclose(r.kappa[2:-2], 1 / 30, rtol=0.01)
    assert abs(r.psi[-1] - np.pi / 2) < 1e-3 and np.allclose([r.x[-1], r.y[-1]], [30, 30], atol=0.05)
    assert np.all(RouteBuilder.arc(30.0, -np.pi / 2, 13.89).kappa[2:-2] < 0)


def test_concat_continuous():
    r = RouteBuilder.concat(
        RouteBuilder.straight(50, 13.89),
        RouteBuilder.arc(30, np.pi / 2, 13.89),
        RouteBuilder.straight(50, 13.89),
    )
    steps = np.hypot(np.diff(r.x), np.diff(r.y))
    assert np.all(steps <= 1.0 + 1e-9) and np.all(np.abs(np.diff(r.psi)) <= 1.0 / 30 + 1e-6)
    assert abs(r.length - (100 + 15 * np.pi)) < 0.1


def test_circle_loop_unwrapped():
    r = RouteBuilder.circle_loop(30.0, 2.0, 10.0)
    assert abs(r.psi[-1] - 4 * np.pi) < 0.01 and np.all(np.diff(r.psi) > 0)


def test_from_waypoints_curvature():
    th = np.linspace(0, np.pi, 200)
    r = RouteBuilder.from_waypoints(np.column_stack([50 * np.sin(th), 50 - 50 * np.cos(th)]), 10.0, 3.5)
    np.testing.assert_allclose(r.kappa[5:-5], 1 / 50, rtol=0.02)
