from dataclasses import replace

import numpy as np
import pytest

from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.dynamics import rk4_step
from hybrid_av.vehicle.identify import (
    LongSamples,
    Segment,
    a_lat_limit,
    check_units,
    fit_coastdown,
    fit_lateral,
    fit_longitudinal,
    friction_scale_for,
    lowpass,
    wheelbase_split,
)
from hybrid_av.vehicle.params import VehicleParams


def simulate(p, vx0, deltas, dt=0.05):
    x = np.array([vx0, 0, 0, 0, 0, 0.0])
    rows = []
    for d in deltas:
        rows.append(x.copy())
        x = rk4_step(x, np.array([d, 0.0]), p, dt)
    a = np.array(rows)
    return Segment(t=np.arange(len(deltas)) * dt, vx=a[:, 0], vy=a[:, 1], r=a[:, 2], delta=np.asarray(deltas))


def test_recovers_known_parameters():
    truth = replace(VehicleParams(), Cf=24000.0, Cr=28000.0, Iz=2600.0)
    segs = [simulate(truth, v, [0.0] * 10 + [s] * 70) for v in (8, 12, 16) for s in (0.03, -0.06, 0.1)]
    fit, info = fit_lateral(segs, VehicleParams())
    assert fit.Cf == pytest.approx(24000, rel=0.1)
    assert fit.Cr == pytest.approx(28000, rel=0.1)
    assert fit.Iz == pytest.approx(2600, rel=0.15)
    assert info["r_rmse"] < 0.02 and info["understeer_K"] > 0


def test_friction_scale_interpolation_and_monotonicity():
    curve = {1.0: 8.5, 0.5: 5.0, 0.4: 4.0, 0.3: 3.0}
    assert friction_scale_for(curve, 0.35 * 9.81) == pytest.approx(0.343, abs=0.01)
    with pytest.raises(ValueError):
        friction_scale_for({1.0: 8.0, 0.5: 9.0}, 3.4)
    with pytest.raises(ValueError):  # target outside the measured range
        friction_scale_for(curve, 20.0)


def test_wheelbase_split_and_unit_check():
    # CARLA frame, ego at (100, 50) heading +y (yaw 90): wheel positions are world cm
    front = [(10000 + 80, 5000 + 140), (10000 - 80, 5000 + 140)]
    rear = [(10000 + 80, 5000 - 148), (10000 - 80, 5000 - 148)]
    L, a, b = wheelbase_split(front, rear, com_x=0.1, loc=(100.0, 50.0), yaw_deg=90.0)
    assert L == pytest.approx(2.88) and a == pytest.approx(1.30) and b == pytest.approx(1.58)
    check_units(L, 0.1)
    check_units(3.005, 0.45)  # CARLA 0.9.16 vehicle.tesla.model3, measured on the pod (G2)
    with pytest.raises(ValueError, match="ECC-32"):
        check_units(288.0, 0.1)
    with pytest.raises(ValueError, match="ECC-32"):
        check_units(2.9, 3.0)


def test_lowpass_attenuates_noise_keeps_mean():
    x = np.full(200, 2.0) + np.random.default_rng(0).normal(0, 1.0, 200)
    y = lowpass(x, 0.05, 5.0)
    assert np.std(y[50:]) < np.std(x[50:]) and np.mean(y[50:]) == pytest.approx(2.0, abs=0.3)


def test_fit_longitudinal_and_coastdown_recover_map():
    rng = np.random.default_rng(1)
    p = VehicleParams()
    vx = rng.uniform(3.0, 20.0, 400)
    ax = rng.uniform(-3.0, 2.0, 400)
    need = ax + p.rolling + p.drag_coeff * vx**2
    throttle = np.where(need > 0, 0.12 + 0.30 * need, 0.0)
    brake = np.where(need <= 0, 0.25 * -need, 0.0)
    s = LongSamples(vx, ax, np.clip(throttle, 0, 1), np.clip(brake, 0, 1))
    lm = fit_longitudinal(s, p)
    assert isinstance(lm, LongitudinalMap)
    assert lm.throttle_gain == pytest.approx(0.30, rel=0.05) and lm.throttle_bias == pytest.approx(
        0.12, abs=0.03
    )
    assert lm.brake_gain == pytest.approx(0.25, rel=0.05)
    v = np.linspace(20, 6, 100)
    rolling, drag = fit_coastdown(v, -(0.2 + 0.0006 * v**2))
    assert rolling == pytest.approx(0.2, abs=1e-6) and drag == pytest.approx(0.0006, abs=1e-8)


def test_a_lat_limit_sustained_before_exceedance():
    dt = 0.05
    t = np.arange(0, 40, dt)
    a = 0.2 * t  # ramp
    lateral = np.where(t < 30, 0.1, 1.5)  # leaves the circle at t = 30 s
    slip = np.zeros_like(t)
    lim = a_lat_limit(a, lateral, slip, dt)
    assert lim == pytest.approx(0.2 * 28.0, abs=0.1)  # highest value held for 2 s before t = 30
    slip[400] = 0.2  # sliding at t = 20 s ends the usable run earlier
    assert a_lat_limit(a, lateral, slip, dt) == pytest.approx(0.2 * 18.0, abs=0.1)
