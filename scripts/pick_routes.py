"""Pick nominal CARLA start/goal routes (G2 Task 7 Step 2): 300-800 m, no lane change (ECC-13), start/goal >= 50 m apart.

pick_routes.py --towns Town01,Town02,Town03_Opt,Town05 --n 10 --set train --seed 0 --out OUT.yaml
(then `--towns Town10HD --set town10hd`). Needs a CARLA server (CARLA_PORT honoured). The YAML entries match the
`{town, start: [x, y, z, yaw_deg], goal: [x, y, z]}` format of configs/scenarios/carla_routes.yaml.
"""

import argparse
import itertools
import math
import random
import sys
from pathlib import Path

import yaml

from hybrid_av.sim.carla_backend.client import CarlaSimConfig, connect
from hybrid_av.sim.carla_backend.scenario_setup import trace_waypoints

LENGTH_M = (300.0, 800.0)
MIN_SEPARATION_M = 50.0
MAX_ATTEMPTS = 400  # (start, goal) pairs tried per town


def _route_length(wps) -> float:
    locs = [w.transform.location for w in wps]
    return sum(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) for a, b in itertools.pairwise(locs))


def town_routes(town: str, seed: int):
    """Yield accepted routes for one town, stopping after MAX_ATTEMPTS pairs."""
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    _, world = connect(CarlaSimConfig(town=town))
    wmap = world.get_map()
    pts = list(wmap.get_spawn_points())
    random.Random(seed).shuffle(pts)
    planner = GlobalRoutePlanner(wmap, 2.0)
    n = len(pts)
    for k in range(min(MAX_ATTEMPTS, n * (n - 1))):
        s, g = pts[k % n], pts[(k % n + 1 + k // n) % n]
        if math.dist((s.location.x, s.location.y), (g.location.x, g.location.y)) < MIN_SEPARATION_M:
            continue
        try:  # the simulator's own checks: lane change, backward waypoints, start snap
            wps = trace_waypoints(
                wmap,
                [s.location.x, s.location.y, s.location.z, s.rotation.yaw],
                [g.location.x, g.location.y, g.location.z],
                planner=planner,
            )
        except ValueError:
            continue
        length = _route_length(wps)
        if LENGTH_M[0] <= length <= LENGTH_M[1]:
            r = lambda *v: [round(float(x), 2) for x in v]
            yield {
                "town": town,
                "start": r(s.location.x, s.location.y, s.location.z, s.rotation.yaw),
                "goal": r(g.location.x, g.location.y, g.location.z),
                "length_m": round(length, 2),
            }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--towns", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--set", dest="set_name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    gens = {t: town_routes(t, a.seed) for t in a.towns.split(",")}
    routes = []
    while gens and len(routes) < a.n:  # round-robin; a town drops out when its attempts run out
        for t in list(gens):
            r = next(gens[t], None)
            if r is None:
                del gens[t]
            elif len(routes) < a.n:
                routes.append(r)
                print(f"{r['town']} start={r['start']} goal={r['goal']} length={r['length_m']} m")
    a.out.write_text(yaml.safe_dump({"set": a.set_name, "routes": routes}, sort_keys=False))
    if len(routes) < a.n:
        towns = a.towns.split(",")
        share = math.ceil(a.n / len(towns))  # a town is short if it gave less than its fair share
        short = [t for t in towns if sum(r["town"] == t for r in routes) < share]
        print(
            f"only {len(routes)}/{a.n} routes found; towns that ran short: {', '.join(short)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
