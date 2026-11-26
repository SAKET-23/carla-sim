from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import FrictionZone, Scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.types import ActuatorCommand
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams


def _scn(route, mu=1.0, zones=(), v0=10.0):
    return Scenario("t", 0, route, None, zones, mu, (), (), (), 600.0, v0)


def test_coasting_decelerates():
    sim = LiteSim(VehicleParams(), LongitudinalMap())
    obs = sim.reset(_scn(RouteBuilder.straight(500, 13.89)))
    for _ in range(40):
        obs, _ = sim.step(ActuatorCommand(0.0, 0.0, 0.0))
    assert 9.0 < obs.ego.vx < 10.0 and abs(obs.ego.y) < 1e-9


def test_skidpad_lateral_limit():
    for mu in (1.0, 0.35):
        sim = LiteSim(VehicleParams(), LongitudinalMap())
        sim.reset(_scn(RouteBuilder.straight(5000, 30), mu=mu, v0=5.0))
        lat = []
        for _ in range(600):  # constant left steer, gentle throttle → rising speed
            obs, _ = sim.step(ActuatorCommand(-0.3, 0.4, 0.0))
            lat.append(abs(obs.ego.vx * obs.ego.r))
        assert max(lat[100:]) <= 1.05 * mu * 9.81
    assert max(abs(sim.ego_state().vy), 0) > 1.0  # at μ = 0.35 the car slides


def test_steering_rate_limited():
    sim = LiteSim(VehicleParams(), LongitudinalMap())
    sim.reset(_scn(RouteBuilder.straight(500, 13.89)))
    prev = sim.steer_angle()
    for _ in range(10):
        sim.step(ActuatorCommand(-1.0, 0.0, 0.0))
        cur = sim.steer_angle()
        assert cur - prev <= 3.0 * 0.05 + 1e-9
        prev = cur  # 3.0 rad/s wheel rate (simulation-backends.md §1) = MPC ddelta_max 0.15/step


def test_friction_zone_switch():
    sim = LiteSim(VehicleParams(), LongitudinalMap())
    obs = sim.reset(_scn(RouteBuilder.straight(500, 13.89), zones=(FrictionZone(20.0, 60.0, 0.35),)))
    assert obs.friction_true == 1.0 and sim.mu_at(30.0) == 0.35 and sim.mu_at(60.1) == 1.0
    while obs.ego.x < 21.0:
        obs, _ = sim.step(ActuatorCommand(0.0, 0.2, 0.0))
    assert obs.friction_true == 0.35


def test_deterministic():
    def run():
        sim = LiteSim(VehicleParams(), LongitudinalMap())
        sim.reset(_scn(RouteBuilder.straight(500, 13.89)))
        return [sim.step(ActuatorCommand(-0.2, 0.3, 0.0))[0].ego.y for _ in range(50)]

    assert run() == run()


def test_brake_stops_at_zero():
    sim = LiteSim(VehicleParams(), LongitudinalMap())
    sim.reset(_scn(RouteBuilder.straight(500, 13.89), v0=3.0))
    for _ in range(100):
        obs, _ = sim.step(ActuatorCommand(0.0, 0.0, 1.0))
        assert obs.ego.vx >= -1e-9
    e = obs.ego
    assert abs(e.vx) < 1e-9 and abs(e.vy) < 1e-3 and abs(e.r) < 1e-3
