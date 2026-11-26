"""Find the CARLA tire-friction scale that gives an effective mu (Plan/06-integration/calibration.md §3).

calibrate_friction.py --target-mu 0.35 --out configs/scenarios/wet_curve.yaml
Drives the MPC around a circle of R = 30 m in the fixture's open area with the speed reference ramped at 0.1 m/s per
second, once per friction scale k, and writes `carla_tire_friction_scale: k*` into --out. Needs a CARLA server.
"""

import argparse
import dataclasses
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import yaml

from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.planning.route import RouteBuilder, place_route
from hybrid_av.sim.base import FrictionZone
from hybrid_av.sim.carla_backend import convert
from hybrid_av.sim.carla_backend.client import CarlaSimConfig
from hybrid_av.sim.carla_backend.routes import plain_scenario
from hybrid_av.sim.carla_backend.sim import CarlaSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import PolicyAction
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.identify import G, a_lat_limit, friction_scale_for
from hybrid_av.vehicle.params import VehicleParams

FIXTURE = Path(__file__).resolve().parents[1] / "tests/integration/fixtures/town03_points.yaml"
RADIUS, V0, V_MAX, RAMP = 30.0, 5.0, 17.0, 0.1  # m, m/s, m/s, m/s per second
SCALES = (1.0, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2)
KEY = "carla_tire_friction_scale"


def circle_start(points: dict, town: str) -> dict:
    """carla_route whose start is on a left-turning circle of R = 30 m centred on the open area; the goal is a road
    point (the planner needs one; the circle itself is not a road)."""
    cx, cy, cz = points["open_area"]
    lx, ly, _ = convert.loc_to_lib(cx, cy, cz)
    sx, sy, sz = convert.lib_to_loc(
        lx, ly - RADIUS, cz
    )  # library heading +x, the circle centre is on the left
    gx, gy, gz, _ = points["straight"]
    return {"town": town, "start": [sx, sy, sz, 0.0], "goal": [gx, gy, gz]}


def circle_run(sim, rc, k, params, lm) -> float:
    """a_lat_max (m/s^2) at friction scale k."""
    # ECC-17: a FrictionZone covering the whole route applies k x the base tire_friction (the scale replaces mu)
    obs = sim.reset(plain_scenario(rc, V0, (FrictionZone(-1e9, 1e9, k),)))
    e = obs.ego
    loop = RouteBuilder.circle_loop(RADIUS, 8.0, V_MAX)
    speed = np.minimum(np.sqrt(V0**2 + 2 * RAMP * loop.s), V_MAX)  # v dv/ds = ramp
    route = place_route(dataclasses.replace(loop, speed_limit=speed), e.x, e.y, e.psi)
    # the reference and the MPC friction margin must not cap the speed below the tyre limit
    stack = AutonomyStack(
        params, lm, dataclasses.replace(MPCConfig(), friction_margin=10.0), ReferenceConfig(a_lat_max=1e3)
    )
    stack.reset(route)
    a_lat, lateral, slip, over = [], [], [], 0
    for _ in range(round(200.0 / sim.dt)):
        feats = stack.features(obs)
        out = stack.act(obs, PolicyAction(1.0, 0.0, 0.0), feats)
        e = obs.ego
        a_lat.append(e.vx * e.r)
        lateral.append(feats.lateral)
        slip.append(e.vy / max(e.vx, 1.0))
        obs, _ = sim.step(out.command)
        over += abs(lateral[-1]) > 1.0 or abs(slip[-1]) > 0.1
        if over > 20 or feats.route_s > route.length - 5.0:  # one second past the limit, or out of route
            break
    return a_lat_limit(np.array(a_lat), np.array(lateral), np.array(slip), sim.dt)


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
    ap.add_argument("--fixture", type=Path, default=FIXTURE)
    ap.add_argument("--town", default="Town03")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--runs", type=Path, default=Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}_calibration"
    )
    a = ap.parse_args(argv)
    params = VehicleParams.from_yaml(a.vehicle) if a.vehicle.exists() else VehicleParams()
    lm = LongitudinalMap.from_yaml(a.longitudinal_map) if a.longitudinal_map.exists() else LongitudinalMap()
    rc = circle_start(yaml.safe_load(a.fixture.read_text()), a.town)
    a.runs.mkdir(parents=True, exist_ok=True)
    sim = CarlaSim(CarlaSimConfig(town=a.town, sensors=None, no_rendering=True))
    try:
        curve = {k: circle_run(sim, rc, k, params, lm) for k in SCALES}
        print(
            "a_lat_max(k):",
            {k: round(v, 2) for k, v in curve.items()},
            f"dry effective mu {curve[1.0] / G:.2f}",
        )
        k_star = friction_scale_for(curve, a.target_mu * G)  # raises if the curve is not monotone
        repeats = [circle_run(sim, rc, k_star, params, lm) for _ in range(a.repeats)]
    finally:
        sim.close()
    spread = (max(repeats) - min(repeats)) / (2 * np.mean(repeats))
    print(f"k* = {k_star:.4f}; repeats {np.round(repeats, 2).tolist()} (+/-{spread:.1%}, want < 5 %)")
    set_key(a.out, KEY, k_star)
    (a.runs / "friction.json").write_text(
        json.dumps({"curve": curve, "k_star": k_star, "repeats": repeats}, indent=2)
    )
    # shortcut: the 9.4 m/s hold / 11 m/s slide discrimination check is not automated
    if spread >= 0.05 or math.isnan(spread):
        raise SystemExit("repeats disagree by more than 5 %: do not trust k*")


if __name__ == "__main__":
    main()
