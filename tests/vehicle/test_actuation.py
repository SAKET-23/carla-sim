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
    accels = tuple(tuple(6.0 * t**2 * (1 - 0.8 * v / 25.0) - 0.38 * v for t in thr) for v in speeds)
    return LongitudinalMap(brake_gain=0.2, table_speeds=speeds, table_throttles=thr, table_accels=accels)


def test_table_inverts_ignores_resistances_and_linear_default_unchanged():
    lm = _table_lm()
    for vx in (2.0, 10.0, 20.0):
        for t in (0.6, 0.8, 0.9):
            ax = 6.0 * t**2 * (1 - 0.8 * vx / 25.0) - 0.38 * vx
            assert np.isclose(control_to_actuator(Control(0.0, ax), vx, P, lm).throttle, t, atol=0.02)
    assert control_to_actuator(Control(0.0, 99.0), 10.0, P, lm).throttle == 1.0
    assert np.isclose(control_to_actuator(Control(0.0, 2.0), 10.0, P, LM).throttle, 0.10 + 0.35 * 2.19)


def test_table_brakes_below_coast_accel_and_coasts_inside_band():
    lm = _table_lm()  # coast accel at 10 m/s is -3.8
    c = control_to_actuator(Control(0.0, -3.9), 10.0, P, lm)
    assert c.throttle == 0.0 and c.brake == 0.0
    c = control_to_actuator(Control(0.0, -6.8), 10.0, P, lm)
    assert c.throttle == 0.0 and np.isclose(c.brake, 0.2 * 3.0)


def test_table_yaml_round_trip(tmp_path):
    import yaml

    from hybrid_av.config import load_dataclass

    lm = _table_lm()
    f = tmp_path / "lm.yaml"
    f.write_text(yaml.safe_dump(__import__("dataclasses").asdict(lm)))
    assert load_dataclass(LongitudinalMap, f) == lm


def _low_lm():  # lowest node 0.77 m/s with coast accel -0.43 (the CARLA calibrated table's shape)
    thr = (0.0, 0.1, 0.5, 1.0)
    return LongitudinalMap(
        table_speeds=(0.77, 10.0),
        table_throttles=thr,
        table_accels=((-0.43, -0.12, 2.0, 5.0), (-3.8, -3.0, 0.5, 3.0)),
    )


def test_table_standstill_holds_instead_of_creeping():
    c = control_to_actuator(Control(0.0, -0.2), 0.09, P, _low_lm())
    assert c.throttle == 0.0 and c.brake > 0.0


def test_table_low_speed_launch_and_mild_decel_still_throttle():
    lm = _low_lm()
    assert control_to_actuator(Control(0.0, 0.5), 0.1, P, lm).throttle > 0.0
    assert 0.0 < control_to_actuator(Control(0.0, -3.0), 10.0, P, lm).throttle < 0.2
