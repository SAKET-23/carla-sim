import numpy as np

from hybrid_av.features.friction import FrictionEstimator
from hybrid_av.planning.route import RouteBuilder
from hybrid_av.sim.base import Scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.types import ActuatorCommand, EgoState
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams


def _scn(route, mu=1.0, zones=(), v0=10.0):
    return Scenario("t", 0, route, None, zones, mu, (), (), (), 600.0, v0)


def _drive(mu, steps=600, steer=-0.3, throttle=0.4):
    sim = LiteSim(VehicleParams(), LongitudinalMap())
    sim.reset(_scn(RouteBuilder.straight(5000, 30), mu=mu, v0=5.0))
    est, out = FrictionEstimator(VehicleParams()), []
    for _ in range(steps):
        obs, _ = sim.step(ActuatorCommand(steer, throttle, 0.0))
        out.append(est.update(obs.ego, sim.steer_angle(), 0.05))
    return np.array(out), sim


def test_straight_no_excitation_holds_prior():
    mu_hat, _ = _drive(1.0, steer=0.0)
    np.testing.assert_allclose(mu_hat, 1.0, atol=1e-9)


def test_dry_skidpad_stays_high():
    mu_hat, _ = _drive(1.0)
    assert mu_hat.min() >= 0.9


def test_wet_slide_detected_within_1s():
    mu_hat, _ = _drive(0.35)
    onset = np.argmax(mu_hat < 0.99)  # first step the estimate moves
    assert onset > 0 and mu_hat[min(onset + 20, len(mu_hat) - 1)] <= 0.5


def test_clamped():
    est = FrictionEstimator(VehicleParams(), mu_init=5.0)
    assert est.update(EgoState(0, 0, 0, 0, 10, 0, 0), 0.0, 0.05) == 1.2
