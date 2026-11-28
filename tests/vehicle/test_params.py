import pytest

from hybrid_av.vehicle.params import VehicleParams


def test_defaults_and_wheelbase():
    p = VehicleParams()
    assert (p.m, p.Iz, p.a, p.b, p.Cf, p.Cr, p.max_steer) == (
        1500.0,
        2250.0,
        1.2,
        1.6,
        19000.0,
        20000.0,
        0.70,
    )
    assert p.L == 2.8


def test_from_yaml_matches_defaults():
    assert VehicleParams.from_yaml("configs/vehicle/default.yaml") == VehicleParams()


def test_committed_carla_model3():
    import numpy as np

    p = VehicleParams.from_yaml("configs/vehicle/carla_model3.yaml")
    assert abs(p.a + p.b - 3.005) < 0.01
    assert len(p.steer_curve_v) > 1 and len(p.steer_curve_v) == len(p.steer_curve_k)
    assert np.all(np.diff(p.steer_curve_k) <= 0)
    assert p.max_steer_wheel == pytest.approx(1.222, abs=0.001)


def test_committed_carla_longitudinal():
    from hybrid_av.types import Control
    from hybrid_av.vehicle.actuation import LongitudinalMap, control_to_actuator

    p = VehicleParams.from_yaml("configs/vehicle/carla_model3.yaml")
    lm = LongitudinalMap.from_yaml("configs/vehicle/carla_model3_longitudinal.yaml")
    assert lm.table_speeds and lm.table_accels
    c = control_to_actuator(Control(delta=0.0, ax=-6.0), 10.0, p, lm)
    assert c.brake > 0 and c.throttle == 0
    assert control_to_actuator(Control(delta=0.0, ax=1.0), 10.0, p, lm).throttle > 0
