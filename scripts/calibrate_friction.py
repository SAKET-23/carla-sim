"""Find the CARLA tire-friction scale that gives an effective mu (Plan/06-integration/calibration.md §3).

calibrate_friction.py --target-mu 0.35 --out configs/scenarios/wet_curve.yaml
Drives the MPC along a real road curve (R = 32.6 m, 80 deg, Town03_Opt by default) once per constant target speed
(6..17 m/s) and friction scale k, and writes `carla_tire_friction_scale: k*` into --out. A pass slides when the
cross-track error exceeds 1 m in the curve or the episode ends off-road or in a collision. Needs a CARLA server.
"""

import argparse
import dataclasses
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.sim.base import FrictionZone
from hybrid_av.sim.carla_backend.client import CarlaSimConfig
from hybrid_av.sim.carla_backend.routes import plain_scenario
from hybrid_av.sim.carla_backend.sim import CarlaSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import PolicyAction
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.identify import G, friction_scale_for
from hybrid_av.vehicle.params import VehicleParams

CURVE = "-11.97,-204.81,0.0,-178.6:-84.97,-151.12,0.0"  # start x,y,z,yaw : goal x,y,z (CARLA coords)
CURVE_R, CURVE_KAPPA, MAX_CROSS = 32.6, 1 / 40, 1.0  # m, 1/m (curve section), m
SPEEDS = tuple(4.0 + 0.5 * i for i in range(27))  # m/s, 4.0 ... 17.0
# pod G2: k >= 0.3 never slid below the 17 m/s cap on R 32.6 m (a_lat_max there is v^2/R, not grip); 0.2 slid at 15 m/s
SCALES = (0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08)
KEY = "carla_tire_friction_scale"


def parse_curve(text: str, town: str) -> dict:
    """carla_route from `START_X,START_Y,START_Z,YAW:GOAL_X,GOAL_Y,GOAL_Z`."""
    start, goal = (list(map(float, part.split(","))) for part in text.split(":"))
    if len(start) != 4 or len(goal) != 3:
        raise SystemExit("--curve must be START_X,START_Y,START_Z,YAW:GOAL_X,GOAL_Y,GOAL_Z")
    return {"town": town, "start": start, "goal": goal}


def curve_pass(sim, rc, k, v, params, lm) -> tuple[float, bool]:
    """One pass at constant speed v and friction scale k: (peak |vx*r| over the curve section, slid)."""
    # ECC-17: a FrictionZone covering the whole route applies k x the base tire_friction (the scale replaces mu)
    obs = sim.reset(plain_scenario(rc, v, (FrictionZone(-1e9, 1e9, k),)))
    r0 = sim.route()
    route = dataclasses.replace(r0, speed_limit=np.full_like(r0.s, v))
    in_curve = np.abs(r0.kappa) > CURVE_KAPPA
    # the reference and the MPC friction margin must not cap the speed below the tyre limit
    stack = AutonomyStack(
        params, lm, dataclasses.replace(MPCConfig(), friction_margin=10.0), ReferenceConfig(a_lat_max=1e3)
    )
    stack.reset(route)
    peak, slid = 0.0, False
    for _ in range(round((route.length / v + 20.0) / sim.dt)):
        feats = stack.features(obs)
        out = stack.act(obs, PolicyAction(1.0, 0.0, 0.0), feats)
        curved = bool(np.interp(feats.route_s, r0.s, in_curve))
        if curved:
            peak = max(peak, abs(obs.ego.vx * obs.ego.r))
        obs, info = sim.step(out.command)
        slid = (curved and abs(feats.lateral) > MAX_CROSS) or info.off_road or info.collision != "none"
        if slid or info.route_completed or info.route_s > route.length - 2.0:
            break
    return peak, slid


def curve_run(sim, rc, k, params, lm) -> tuple[float, float | None]:
    """(a_lat_max, first sliding speed or None) at friction scale k: the largest peak among held passes."""
    a_lat = 0.0
    for v in SPEEDS:
        peak, slid = curve_pass(sim, rc, k, v, params, lm)
        if slid:
            return a_lat, v  # shortcut: the sweep stops at the first slide, assumes sliding is monotone in v
        a_lat = max(a_lat, peak)
    return a_lat, None


def set_key(path: Path, key: str, value: float) -> None:
    text = path.read_text() if path.exists() else ""
    line = f"{key}: {value:.4f}"
    text = (
        re.sub(rf"^{key}:.*$", line, text, flags=re.MULTILINE)
        if re.search(rf"^{key}:", text, re.MULTILINE)
        else text.rstrip("\n") + f"\n{line}\n"
    )
    path.write_text(text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-mu", type=float, default=0.35)
    ap.add_argument(
        "--out", type=Path, required=True, help="scenario YAML that gets `carla_tire_friction_scale`"
    )
    ap.add_argument("--vehicle", type=Path, default=Path("configs/vehicle/carla_model3.yaml"))
    ap.add_argument(
        "--longitudinal-map", type=Path, default=Path("configs/vehicle/carla_model3_longitudinal.yaml")
    )
    ap.add_argument("--curve", default=CURVE, help="START_X,START_Y,START_Z,YAW:GOAL_X,GOAL_Y,GOAL_Z (CARLA)")
    ap.add_argument("--town", default="Town03_Opt")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--runs", type=Path, default=Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_calibration"
    )
    a = ap.parse_args(argv)
    params = VehicleParams.from_yaml(a.vehicle) if a.vehicle.exists() else VehicleParams()
    lm = LongitudinalMap.from_yaml(a.longitudinal_map) if a.longitudinal_map.exists() else LongitudinalMap()
    rc = parse_curve(a.curve, a.town)
    a.runs.mkdir(parents=True, exist_ok=True)
    sim = CarlaSim(CarlaSimConfig(town=a.town, sensors=None, no_rendering=True))
    v_hold = math.sqrt(a.target_mu * G * CURVE_R)
    try:
        runs = {k: curve_run(sim, rc, k, params, lm) for k in SCALES}
        # only grip-limited scales: a scale that never slid measured the speed cap, not the tyres
        curve = {k: r[0] for k, r in runs.items() if r[1] is not None}
        print(
            "a_lat_max(k):",
            {k: round(r[0], 2) for k, r in runs.items()},
            "first slide (m/s):",
            {k: r[1] for k, r in runs.items()},
        )
        if len(curve) < 2:
            raise SystemExit(f"fewer than 2 scales slid below {SPEEDS[-1]} m/s: extend SCALES downward")
        k_star = friction_scale_for(curve, a.target_mu * G)  # raises if the curve is not monotone
        repeats = [curve_run(sim, rc, k_star, params, lm)[0] for _ in range(a.repeats)]
        # at k*, sqrt(target_mu g R) must be held and 1.2x that must slide
        hold = {"v_hold": v_hold, "v_slide": 1.2 * v_hold}
        hold["held"] = not curve_pass(sim, rc, k_star, hold["v_hold"], params, lm)[1]
        hold["slid"] = curve_pass(sim, rc, k_star, hold["v_slide"], params, lm)[1]
    finally:
        sim.close()
    spread = (max(repeats) - min(repeats)) / (2 * np.mean(repeats))
    print(f"k* = {k_star:.4f}; repeats {np.round(repeats, 2).tolist()} (+/-{spread:.1%}, want < 5 %)")
    print(f"hold {v_hold:.1f} m/s: held={hold['held']}; slide {hold['v_slide']:.1f} m/s: slid={hold['slid']}")
    set_key(a.out, KEY, k_star)
    (a.runs / "friction.json").write_text(
        json.dumps(
            {
                "curve": curve,
                "first_slide": {k: r[1] for k, r in runs.items()},
                "k_star": k_star,
                "repeats": repeats,
                "hold": hold,
            },
            indent=2,
        )
    )
    if spread >= 0.05 or math.isnan(spread):
        raise SystemExit("repeats disagree by more than 5 %: do not trust k*")
    if not (hold["held"] and hold["slid"]):
        raise SystemExit(f"k* fails the discrimination check (held/slid wanted True/True): {hold}")


if __name__ == "__main__":
    main()
