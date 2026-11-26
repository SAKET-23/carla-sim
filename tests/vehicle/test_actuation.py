import numpy as np

from hybrid_av.types import Control
from hybrid_av.vehicle.actuation import LongitudinalMap, control_to_actuator
from hybrid_av.vehicle.params import VehicleParams

P, LM = VehicleParams(), LongitudinalMap()


def test_steer_sign_and_clip():
    assert np.isclose(
        control_to_actuator(Control(0.35, 0.0), 10.0, P, LM).steer, -0.5
    )  # left delta → CARLA negative
    assert control_to_actuator(Control(-1.0, 0.0), 10.0, P, LM).steer == 1.0


def test_throttle_value():  # required = 2.0 + 0.15 + 0.0004*100 = 2.19
    c = control_to_actuator(Control(0.0, 2.0), 10.0, P, LM)
    assert np.isclose(c.throttle, 0.10 + 0.35 * 2.19) and c.brake == 0.0


def test_brake_value():  # required = -3.0 + 0.19 = -2.81
    c = control_to_actuator(Control(0.0, -3.0), 10.0, P, LM)
    assert np.isclose(c.brake, 0.20 * 2.81) and c.throttle == 0.0


def test_coast_band_and_exclusive():
    c = control_to_actuator(Control(0.0, 0.1), 10.0, P, LM)
    assert c.throttle == 0.0 and c.brake == 0.0
    for ax in np.linspace(-4.5, 2.5, 50):
        c = control_to_actuator(Control(0.0, ax), 10.0, P, LM)
        assert not (c.throttle > 0 and c.brake > 0) and 0 <= c.throttle <= 1 and 0 <= c.brake <= 1
