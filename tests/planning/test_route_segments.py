import numpy as np
import pytest

from hybrid_av.planning.route import RouteBuilder, constant_radius_segments, place_route


def test_finds_the_30m_arc():
    r = RouteBuilder.concat(
        RouteBuilder.straight(100, 13.9),
        RouteBuilder.arc(30.0, 1.5708, 13.9),
        RouteBuilder.straight(100, 13.9),
    )
    (seg,) = constant_radius_segments(r, 28.0, 35.0, min_length=20.0)
    assert abs(seg[2] - 30.0) < 1.0 and abs((seg[1] - seg[0]) - 47.1) < 3.0


def test_short_curves_are_ignored():
    r = RouteBuilder.concat(RouteBuilder.straight(50, 13.9), RouteBuilder.arc(30.0, 0.3, 13.9))
    assert constant_radius_segments(r, 28.0, 35.0, min_length=20.0) == []


def test_place_route_rotates_and_translates():
    r = place_route(RouteBuilder.arc(30.0, 1.0, 10.0), x0=5.0, y0=-2.0, psi0=np.pi / 2)
    assert (r.x[0], r.y[0]) == pytest.approx((5.0, -2.0)) and r.psi[0] == pytest.approx(np.pi / 2)
    assert r.x[-1] < 5.0  # a left arc from heading +y bends toward -x
