"""Identify the CARLA ego (Plan/06-integration/calibration.md §1 lateral, §2 longitudinal).

Lateral:       identify_vehicle.py --out configs/vehicle/carla_model3.yaml
Longitudinal:  identify_vehicle.py --longitudinal --vehicle configs/vehicle/carla_model3.yaml \\
                   --out configs/vehicle/carla_model3_longitudinal.yaml
`--longitudinal` writes the LongitudinalMap to --out and updates rolling, drag_coeff, max_accel and max_decel in --vehicle
(VehicleParams YAML files take no extra sections). Needs a CARLA server and the fixture from find_map_features.py.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import yaml

from hybrid_av.sim.carla_backend.client import CarlaSimConfig
from hybrid_av.sim.carla_backend.routes import plain_scenario, straight_route
from hybrid_av.sim.carla_backend.sim import CarlaSim
from hybrid_av.types import ActuatorCommand, Control
from hybrid_av.vehicle.actuation import LongitudinalMap, control_to_actuator
from hybrid_av.vehicle.identify import (
    LongSamples,
    Segment,
    check_units,
    evaluate_lateral,
    fit_coastdown,
    fit_lateral,
    fit_longitudinal,
    invalid_reason,
    lateral_problems,
    long_run_arrays,
    longitudinal_problems,
    steady_gain_error,
    throttle_r2,
    understeer_gradient,
    wheelbase_split,
)
from hybrid_av.vehicle.params import VehicleParams

FIXTURE = Path(__file__).resolve().parents[1] / "tests/integration/fixtures/town03_points.yaml"
KP, KI = 1.0, 0.3  # speed-hold PI, m/s^2 per m/s and per m (the throttle comes from the LongitudinalMap)
STEP_SPEEDS, STEP_ALATS = (8.0, 12.0, 16.0), (1.0, 2.0, 3.0)  # m/s, m/s^2 (linear-tyre regime)
STEP_T = 3.5  # s per step segment: step at 1 s, steady state well before the last 1 s the gain uses
SWEEP_AMP_ALAT = 3.0  # m/s^2 cap of the chirp amplitude at 12 m/s
HOLD_OUT = 16.0  # m/s step segments kept out of the fit
MIN_HELD_OUT = 3  # valid held-out segments needed for a verdict
SWEEP_CHUNKS, SWEEP_T, SWEEP_F = (
    3,
    12.0,
    (0.1, 2.0),
)  # chunks of the 0.1 -> 2 Hz chirp (the straight is ~200 m)


def drive(sim, rc, v0, seconds, delta_of_t, v_ref, params, lm) -> Segment:
    """Reset at v0 and hold v_ref with a PI on throttle while steering delta_of_t(t); records the lateral states."""
    obs = sim.reset(plain_scenario(rc, v0))
    rows, i_err, collided = [], 0.0, False
    for k in range(round(seconds / sim.dt)):
        e = obs.ego
        delta = delta_of_t(k * sim.dt)
        i_err += (v_ref - e.vx) * sim.dt
        cmd = control_to_actuator(Control(delta, KP * (v_ref - e.vx) + KI * i_err), e.vx, params, lm)
        rows.append((e.t, e.vx, e.vy, e.r, delta))
        obs, info = sim.step(cmd)
        if info.collision != "none":
            collided = True
            break
        if info.route_completed:
            break
    a = np.array(rows)
    return Segment(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], collided)


def step_deltas(L: float, v: float) -> list[float]:
    """Step angles (rad, both signs) giving the target lateral accelerations: delta = a_lat L / v^2."""
    return [sg * a * L / v**2 for a in STEP_ALATS for sg in (1.0, -1.0)]


@contextlib.contextmanager
def obstacles_unloaded(sim):
    """Unload the roadside layers so excursions off the lane do not collide; reload them on exit."""
    import carla

    layers = (
        carla.MapLayer.Buildings
        | carla.MapLayer.Props
        | carla.MapLayer.StreetLights
        | carla.MapLayer.Walls
        | carla.MapLayer.Foliage
        | carla.MapLayer.ParkedVehicles
    )  # fmt: skip  (not Ground: the car stays on terrain)
    sim.world.unload_map_layer(layers)
    try:
        yield
    finally:
        sim.world.load_map_layer(layers)


def chirp(f0, f1, T, amp=0.05):
    """delta(t) = amp sin(phase) with the frequency rising linearly from f0 to f1 over T s, after 1 s of zero."""
    return lambda t: (
        0.0 if t < 1.0 else amp * math.sin(2 * math.pi * (f0 * (t - 1) + (f1 - f0) * (t - 1) ** 2 / (2 * T)))
    )


def read_api(sim) -> VehicleParams:
    """Parameters read directly from the API (calibration.md §1 table), after the ECC-32 unit check."""
    ego, phys = sim.ego, sim.ego.get_physics_control()
    loc, tf = ego.get_location(), ego.get_transform()
    w = [(wh.position.x, wh.position.y) for wh in phys.wheels]
    # ECC-32: wheels are ordered FL, FR, RL, RR
    L, a, b = wheelbase_split(w[:2], w[2:], phys.center_of_mass.x, (loc.x, loc.y), tf.rotation.yaw)
    check_units(L, phys.center_of_mass.x)
    ext = ego.bounding_box.extent
    m = float(phys.mass)
    return dataclasses.replace(
        VehicleParams(), m=m, a=a, b=b, Iz=m * a * b, **sim.steer_params,
        max_steer=min(0.70, sim.max_steer), length=2 * ext.x, width=2 * ext.y,
    )  # fmt: skip


def steering_rate(sim, rc) -> float:
    """rad/s of the front wheel after a steer step 0 -> 0.5 at 10 m/s (calibration.md §1 table)."""
    import carla

    sim.reset(plain_scenario(rc, 10.0))
    angles = []
    for k in range(30):
        sim.step(ActuatorCommand(0.5 if k >= 5 else 0.0, 0.3, 0.0))
        # ECC-05, ECC-42: wheel angle in degrees from get_wheel_steer_angle(FL_Wheel)
        angles.append(math.radians(sim.ego.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)))
    return float(np.max(np.abs(np.diff(angles))) / sim.dt)


def _save(runs: Path, segments, summary: dict) -> None:
    (runs / "identify.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        runs / "segments.npz",
        **{
            f"{i}_{k}": getattr(s, k) for i, s in enumerate(segments) for k in ("t", "vx", "vy", "r", "delta")
        },
    )


def lateral(sim, rc, out: Path, runs: Path) -> None:
    sim.reset(plain_scenario(rc, 0.0))  # spawns the ego the API values are read from
    params = read_api(sim)
    lm = LongitudinalMap()
    rate = steering_rate(sim, rc)
    print(f"API: L={params.L:.3f} a={params.a:.3f} b={params.b:.3f} m={params.m:.0f} "
          f"max_steer_wheel={params.max_steer_wheel:.3f} steering rate {rate:.2f} rad/s\n"
          f"steering curve (m/s -> factor): {list(zip(params.steer_curve_v, params.steer_curve_k))}")  # fmt: skip
    if rate < 3.0:
        print(
            f"steering rate {rate:.2f} < 3.0 rad/s: set MPCConfig.ddelta_max = {rate * sim.dt:.4f} and the LiteSim limit"
        )
    with obstacles_unloaded(sim):
        steps = {
            (v, d): drive(sim, rc, v, STEP_T, lambda t, d=d: 0.0 if t < 1.0 else d, v, params, lm)
            for v in STEP_SPEEDS for d in step_deltas(params.L, v)
        }  # fmt: skip
        amp = SWEEP_AMP_ALAT * params.L / 12.0**2
        f = np.linspace(SWEEP_F[0], SWEEP_F[1], SWEEP_CHUNKS + 1)
        sweeps = [
            drive(sim, rc, 12.0, SWEEP_T + 1.0, chirp(f[i], f[i + 1], SWEEP_T, amp), 12.0, params, lm)
            for i in range(SWEEP_CHUNKS)
        ]
    labelled = [(s, v, f"step v={v:g} delta={d:+.4f}") for (v, d), s in steps.items()] + [
        (s, 12.0, f"sweep {i}") for i, s in enumerate(sweeps)
    ]
    train, held, dropped = [], [], []
    for s, v, name in labelled:
        why = invalid_reason(s, v)
        if why:
            dropped.append(f"{name}: {why}")
        elif name.startswith("step") and v == HOLD_OUT:
            held.append(s)
        else:
            train.append(s)
    print(f"excluded {len(dropped)} invalid segments" + "".join(f"\n  {m}" for m in dropped))
    if len(held) < MIN_HELD_OUT:
        _save(runs, train + held, {"pass": False, "excluded": dropped})
        raise SystemExit(f"only {len(held)} valid held-out segments (< {MIN_HELD_OUT}); {out} NOT written")
    fit, info = fit_lateral(train, params)
    val = evaluate_lateral(fit, held)
    K = understeer_gradient(fit)
    gains = [steady_gain_error(s, fit, K) for s in held]
    problems = lateral_problems(val, gains, K)
    ok = not problems
    print(
        f"fit {info}\nheld-out {val}, gain error max {max(gains):.3f}, K {K:.5f}: {'PASS' if ok else 'FAIL'}"
    )
    for msg in problems:
        print(f"  failed: {msg}")
    if ok:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(dataclasses.asdict(fit)))
    else:
        print(f"{out} NOT written")
    _save(
        runs,
        train + held,
        {
            "fit": info,
            "held_out": val,
            "gain_err": gains,
            "steer_rate": rate,
            "pass": bool(ok),
            "excluded": dropped,
        },
    )
    if not ok:
        raise SystemExit(1)


def run_long(sim, rc, v0, seconds, cmd_of, stop=None):
    """(vx, ax low-passed at 5 Hz, throttle, brake) per tick for one open-loop run."""
    obs = sim.reset(plain_scenario(rc, v0))
    rows = []
    for _ in range(round(seconds / sim.dt)):
        e = obs.ego
        thr, brk = cmd_of(e)
        rows.append((e.vx, e.ax, thr, brk))
        obs, info = sim.step(ActuatorCommand(0.0, thr, brk))
        if info.route_completed or (stop and stop(obs.ego)):
            break
    return long_run_arrays(np.array(rows), sim.dt)


def longitudinal(sim, rc, vehicle: Path, out: Path, runs: Path) -> None:
    p = VehicleParams.from_yaml(vehicle)
    levels = [round(0.1 * i, 1) for i in range(1, 11)]
    parts = [
        run_long(sim, rc, v0, 4.0, lambda e, t=t: (t, 0.0), lambda e: e.vx > 25.0)
        for v0 in (0.0, 5.0, 10.0, 15.0)
        for t in levels
    ]
    parts += [
        run_long(sim, rc, v0, 12.0, lambda e, b=b: (0.0, b), lambda e: e.vx < 0.2)
        for v0 in (10.0, 20.0)
        for b in levels
    ]
    coast = run_long(sim, rc, 20.0, 15.0, lambda e: (0.0, 0.0), lambda e: e.vx < 5.0)
    parts.append(coast)  # also gives the coast band
    cv, ca = coast[0], coast[1]
    rolling, drag = fit_coastdown(cv[cv > 6.0], ca[cv > 6.0])
    p = dataclasses.replace(p, rolling=rolling, drag_coeff=drag)
    s = LongSamples(*(np.concatenate(c) for c in zip(*parts, strict=True)))
    lm = fit_longitudinal(s, p)
    r2 = throttle_r2(s, p, lm)
    full = (s.throttle >= 0.99) & (s.vx < 10.0)
    braking = (s.brake >= 0.99) & (s.vx > 2.0)
    p = dataclasses.replace(
        p, max_accel=float(np.percentile(s.ax[full], 90)), max_decel=float(np.mean(-s.ax[braking]))
    )
    print(
        f"{lm}\nrolling {rolling:.4f} drag {drag:.6f} max_accel {p.max_accel:.2f} max_decel {p.max_decel:.2f} R2 {r2:.3f}"
    )
    problems = longitudinal_problems(p, lm, r2)
    if r2 < 0.9:
        print(
            "throttle fit R2 < 0.9: switch LongitudinalMap to a 2-D table and add an expected-code-changes.md entry"
        )
    if problems:
        raise SystemExit("longitudinal identification FAILED, nothing written: " + "; ".join(problems))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(dataclasses.asdict(lm)))
    vehicle.write_text(yaml.safe_dump(dataclasses.asdict(p)))
    (runs / "longitudinal.json").write_text(json.dumps({"map": dataclasses.asdict(lm), "r2": r2}, indent=2))
    # shortcut: the feed-forward profile acceptance run (0 -> 15 -> 5 -> 12 -> 0 m/s) is not automated; it needs a straight > 250 m


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--longitudinal", action="store_true")
    ap.add_argument("--vehicle", type=Path, default=Path("configs/vehicle/carla_model3.yaml"))
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    ap.add_argument("--town", default="Town03_Opt")
    ap.add_argument(
        "--runs", type=Path, default=Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_calibration"
    )
    a = ap.parse_args(argv)
    rc = straight_route(yaml.safe_load(a.fixture.read_text()), a.town)
    a.runs.mkdir(parents=True, exist_ok=True)
    sim = CarlaSim(CarlaSimConfig(town=a.town, sensors=None, no_rendering=True))
    try:
        if a.longitudinal:
            longitudinal(sim, rc, a.vehicle, a.out, a.runs)
        else:
            lateral(sim, rc, a.out, a.runs)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
