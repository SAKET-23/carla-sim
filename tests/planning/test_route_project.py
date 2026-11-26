import numpy as np

from hybrid_av.planning.route import RouteBuilder


def test_project_straight():
    p = RouteBuilder.straight(100, 13.89).project(50.0, 2.0)
    assert abs(p.s - 50.0) < 1e-6 and abs(p.lateral - 2.0) < 1e-6 and abs(p.heading) < 1e-9


def test_project_arc_inside_is_left():
    r = RouteBuilder.arc(30.0, np.pi / 2, 13.89)
    th = np.pi / 4  # mid-arc, centre (0, 30)
    p = r.project(28 * np.sin(th), 30 - 28 * np.cos(th))
    assert abs(p.lateral - 2.0) < 0.02 and abs(p.s - 30 * th) < 0.05


def test_project_uses_hint_on_loop():
    r = RouteBuilder.circle_loop(30.0, 2.0, 10.0)
    lap = 2 * np.pi * 30
    assert abs(r.project(0.0, 0.0, s_hint=lap + 1.0).s - lap) < 0.05
    assert r.project(0.0, 0.0).s < 0.05


def test_sample_clamps():
    r = RouteBuilder.straight(100, 13.89)
    d = r.sample(np.array([-5.0, 50.0, 150.0]))
    np.testing.assert_allclose(d["x"], [0, 50, 100], atol=1e-9)
    assert set(d) == {"x", "y", "psi", "kappa", "speed_limit", "lane_width"}


def test_project_hint_past_end():
    r = RouteBuilder.straight(50, 13.89)
    for hint in (60.0, 70.0):
        assert abs(r.project(50.0, 0.0, s_hint=hint).s - 50.0) < 1e-6
