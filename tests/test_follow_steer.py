from hybrid_av.planning.route import RouteBuilder
from hybrid_av.types import EgoState
from tests.integration.helpers import follow_steer


def _steer(x, y, psi):
    return follow_steer(RouteBuilder.straight(100.0, 10.0), EgoState(0.0, x, y, psi, 5.0, 0.0, 0.0), 0.5)


def test_follow_steer_sign():
    assert _steer(10.0, 0.0, 0.0) == 0.0
    assert _steer(10.0, 1.0, 0.0) < 0  # left of the route: steer right
    assert _steer(10.0, -1.0, 0.0) > 0
    assert _steer(10.0, 0.0, -0.2) > 0  # pointing right of the route: steer left
    assert _steer(10.0, 50.0, 0.0) == -0.5  # clipped
