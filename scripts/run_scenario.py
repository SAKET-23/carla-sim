"""Run one scenario episode in LiteSim or CARLA; writes trace.npz, metrics.json and episode.png to a run directory."""

import argparse
import dataclasses
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from hybrid_av.config import make_run_dir
from hybrid_av.eval.metrics import steady_state_metrics
from hybrid_av.eval.plots import plot_episode
from hybrid_av.eval.runner import run_episode
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.expert import ConstantAction, ExpertPolicy
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.carla_backend.client import CarlaSimConfig
from hybrid_av.sim.carla_backend.routes import carla_scenario as _carla_scenario
from hybrid_av.sim.carla_backend.sim import CarlaSim
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams

_ROOT = Path(__file__).resolve().parents[1]
ROUTES = _ROOT / "configs/scenarios/carla_routes.yaml"
RIG = _ROOT / "configs/sensors/carla_rig.yaml"


def carla_scenario(sc, family, seed, town, route_set=None, route_index=None):
    try:
        return _carla_scenario(sc, family, seed, town, ROUTES, True, route_set, route_index)
    except LookupError as e:
        raise SystemExit(str(e)) from None


def _gpu_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        return max(float(v) for v in out.split())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return float("nan")


class Measured:
    """Wraps a Sim: ends the episode after `ticks` steps and records the wall-clock tick rate and peak GPU memory."""

    def __init__(self, sim, ticks: int | None):
        self._sim, self._ticks = sim, ticks

    def __getattr__(self, name):  # dt, route, close, ...
        return getattr(self._sim, name)

    def reset(self, scenario):
        self.k, self.mem, self.t0, self.t1 = 0, [], None, None
        return self._sim.reset(scenario)

    def step(self, cmd):
        self.t0 = self.t0 or time.perf_counter()
        obs, info = self._sim.step(cmd)
        self.t1, self.k = time.perf_counter(), self.k + 1
        if self.k % 10 == 1:
            self.mem.append(_gpu_mib())
        if self._ticks and self.k >= self._ticks:
            info = dataclasses.replace(info, timeout=True)
        return obs, info

    @property
    def peak(self) -> float:
        return max((m for m in self.mem if not math.isnan(m)), default=float("nan"))

    def rate(self) -> float:
        return self.k / (self.t1 - self.t0) if self.k else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="nominal", choices=["nominal", "wet_curve", "merge_deadlock"])
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--source", default="constant", help="constant | expert | policy:<path>")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--backend", choices=["litesim", "carla"], default="litesim")
    ap.add_argument("--vehicle", type=Path, help="VehicleParams YAML (default: the built-in defaults)")
    ap.add_argument("--longitudinal-map", type=Path, help="LongitudinalMap YAML (default: the built-in map)")
    ap.add_argument("--town", help="CARLA town (overrides the scenario's)")
    ap.add_argument("--route-set", help="carla_routes.yaml set, e.g. town10hd (key <family>_<set>)")
    ap.add_argument("--route-index", type=int, help="route in the set (default: seed %% set size)")
    ap.add_argument(
        "--quality", choices=["Low", "Epic"], default="Low", help="recorded only; set at server launch"
    )
    ap.add_argument("--ticks", type=int, help="stop after this many steps")
    ap.add_argument(
        "--measure", action="store_true", help="append tick rate and peak GPU memory to <out>/vram.csv"
    )
    ap.add_argument(
        "--mode", choices=["A", "C"], default="C", help="A: no rendering, no sensors; C: full sensor rig"
    )
    a = ap.parse_args()
    if a.source == "constant":
        source = ConstantAction()
    elif a.source == "expert":
        source = ExpertPolicy()
    elif a.source.startswith("policy:"):
        raise SystemExit("policy sources arrive in Phase 3")
    else:
        ap.error(f"unknown source {a.source!r}")
    params = VehicleParams.from_yaml(a.vehicle) if a.vehicle else VehicleParams()
    lm = LongitudinalMap.from_yaml(a.longitudinal_map) if a.longitudinal_map else LongitudinalMap()
    mpc, ref = MPCConfig(), ReferenceConfig()
    sc = make_scenario(a.family, a.seed)
    if a.backend == "carla":
        sc = carla_scenario(sc, a.family, a.seed, a.town, a.route_set, a.route_index)
        sim = CarlaSim(
            CarlaSimConfig(
                town=sc.carla_route["town"],
                quality=a.quality,
                sensors=None if a.mode == "A" else RIG,
                no_rendering=a.mode == "A",
            )
        )
    else:
        sim = LiteSim(params, lm)
    sim = Measured(sim, a.ticks)
    try:
        res = run_episode(sim, sc, AutonomyStack(params, lm, mpc, ref), source)
        route = sim.route()
    finally:
        sim.close()
    if a.measure:
        a.out.mkdir(parents=True, exist_ok=True)
        vram = a.out / "vram.csv"
        if not vram.exists():
            vram.write_text("backend,town,quality,mode,ticks,ticks_per_s,peak_vram_mib\n")
        town = sc.carla_route["town"] if sc.carla_route else ""
        with vram.open("a") as f:
            f.write(f"{a.backend},{town},{a.quality},{a.mode},{sim.k},{sim.rate():.2f},{sim.peak:.0f}\n")
    d = make_run_dir(
        f"{a.family}_{a.seed}",
        {
            "params": params,
            "mpc": mpc,
            "ref": ref,
            "args": {"family": a.family, "seed": a.seed, "source": a.source, "backend": a.backend},
        },
        root=a.out,
    )
    np.savez(d / "trace.npz", **res.trace)
    meta = {k: v for k, v in dataclasses.asdict(res).items() if k != "trace"}
    meta.update(steady_state_metrics(res.trace, sc, sim.dt))
    (d / "metrics.json").write_text(json.dumps(meta, indent=2))
    plot_episode(res, route, d / "episode.png")
    print(
        f"{a.family} seed {a.seed} {a.source}: completion {res.route_completion:.3f} "
        f"infractions { {k: v for k, v in res.infractions.items() if v} } "
        f"duration {res.duration:.1f}s DS {res.driving_score:.1f} -> {d}"
    )


if __name__ == "__main__":
    main()
