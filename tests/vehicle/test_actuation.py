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


def test_steer_curve_identity_default_and_model3_curve():
    assert np.isclose(control_to_actuator(Control(0.07, 0.0), 30.0, P, LM).steer, -0.1)
    # CARLA model3 steering_curve (km/h, factor): (0, 1), (20, .9), (60, .8), (120, .7)
    pc = VehicleParams(steer_curve_v=(0.0, 20 / 3.6, 60 / 3.6, 120 / 3.6), steer_curve_k=(1.0, 0.9, 0.8, 0.7))
    f = np.interp(10.0, pc.steer_curve_v, pc.steer_curve_k)
    assert np.isclose(f, 0.86, atol=0.01)
    assert np.isclose(control_to_actuator(Control(0.07, 0.0), 10.0, pc, LM).steer, -0.1 / f)
    assert np.isclose(control_to_actuator(Control(0.07, 0.0), -10.0, pc, LM).steer, -0.1 / f)
    assert control_to_actuator(Control(-1.0, 0.0), 10.0, pc, LM).steer == 1.0


def test_steer_curve_loads_from_yaml(tmp_path):
    f = tmp_path / "v.yaml"
    f.write_text("steer_curve_v: [0, 5]\nsteer_curve_k: [1, 0.9]\n")
    p = VehicleParams.from_yaml(f)
    assert p.steer_curve_v == (0.0, 5.0) and p.steer_curve_k == (1.0, 0.9)


def _table_lm():
    thr = tuple(k / 10 for k in range(11))
    speeds = (2.0, 10.0, 20.0)
    accels = tuple(tuple(6.0 * t**2 * (1 - 0.8 * v / 25.0) for t in thr) for v in speeds)
    return LongitudinalMap(table_speeds=speeds, table_throttles=thr, table_accels=accels)


def test_table_inverts_round_trip_and_linear_default_unchanged():
    lm = _table_lm()
    for vx in (2.0, 6.0, 10.0, 17.0):
        for t in (0.6, 0.8, 0.9):
            f = np.interp(
                vx, (2.0, 10.0, 20.0), [6.0 * t**2 * (1 - 0.8 * v / 25.0) for v in (2.0, 10.0, 20.0)]
            )
            ax = f - P.rolling - P.drag_coeff * vx**2
            got = control_to_actuator(Control(0.0, ax), vx, P, lm).throttle
            assert np.isclose(got, t, atol=0.05)
    assert control_to_actuator(Control(0.0, 99.0), 10.0, P, lm).throttle == 1.0
    assert np.isclose(control_to_actuator(Control(0.0, 2.0), 10.0, P, LM).throttle, 0.10 + 0.35 * 2.19)


def test_table_yaml_round_trip(tmp_path):
    import yaml

    from hybrid_av.config import load_dataclass

    lm = _table_lm()
    f = tmp_path / "lm.yaml"
    f.write_text(yaml.safe_dump(__import__("dataclasses").asdict(lm)))
    assert load_dataclass(LongitudinalMap, f) == lm
