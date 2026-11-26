"""Closed-loop MPC solve-time benchmark: prints p50/p90/p99/max and writes <run_dir>/metrics.json."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# deliberately reuses the test helpers (tests/mpc/helpers.py) for the benchmark scenarios
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests.mpc.helpers lives under the repo root

from hybrid_av.config import make_run_dir
from hybrid_av.mpc.controller import BicycleMPC, MPCConfig
from hybrid_av.vehicle.params import VehicleParams
from tests.mpc.helpers import circle_reference, closed_loop, straight_reference


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--scenario", choices=["straight", "circle"], default="circle")
    a = ap.parse_args()
    params, cfg = VehicleParams(), MPCConfig()
    if a.scenario == "straight":
        ref_fn = lambda e: straight_reference(e, 10.0)
    else:
        ref_fn = lambda e: circle_reference(e, 40.0, 10.0)
    out = closed_loop(BicycleMPC(params, cfg), ref_fn, np.array([10.0, 0, 0, 0, 0, 0]), a.steps)
    t = out["solve_time"][5:] * 1e3  # skip warm-up solves
    m = {
        "scenario": a.scenario,
        "steps": a.steps,
        "p50_ms": float(np.percentile(t, 50)),
        "p90_ms": float(np.percentile(t, 90)),
        "p99_ms": float(np.percentile(t, 99)),
        "max_ms": float(t.max()),
    }
    m["hz_at_p50"] = 1e3 / m["p50_ms"]
    d = make_run_dir("bench_mpc", {"params": params, "mpc": cfg})
    (d / "metrics.json").write_text(json.dumps(m, indent=2))
    print(
        f"{a.scenario}: p50 {m['p50_ms']:.2f} ms  p90 {m['p90_ms']:.2f} ms  p99 {m['p99_ms']:.2f} ms  "
        f"max {m['max_ms']:.2f} ms  ({m['hz_at_p50']:.0f} Hz at p50)  -> {d}/metrics.json"
    )


if __name__ == "__main__":
    main()
