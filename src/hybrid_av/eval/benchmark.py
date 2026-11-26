"""Methods x scenarios x seeds benchmark with percentile-bootstrap CIs (Plan/07-evaluation/metrics.md section 0)."""

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from hybrid_av.eval.metrics import ood_failed
from hybrid_av.eval.runner import run_episode
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.policy.io import load_policy
from hybrid_av.policy.network import TorchPolicySource
from hybrid_av.rl.expert import ConstantAction, ExpertPolicy
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams

_METRICS = {  # summary key -> row column
    "driving_score": "driving_score",
    "route_completion": "route_completion",
    "failure_rate": "failed",
    "complete_frac": "complete",
}


@dataclass
class BenchmarkReport:
    rows: list[dict]
    summary: dict


def _make_source(method: dict):
    kind = method["source"]
    if kind == "constant":
        return ConstantAction()
    if kind == "expert":
        return ExpertPolicy()
    if kind == "policy":
        return TorchPolicySource(load_policy(Path(method["path"]))[0])
    raise ValueError(f"unknown source {kind!r}")


def _run_one(task: tuple) -> dict:
    method, family, seed, overrides = task
    params, lm = VehicleParams(), LongitudinalMap()
    stack = AutonomyStack(params, lm, MPCConfig(), ReferenceConfig())
    res = run_episode(
        LiteSim(params, lm), make_scenario(family, seed, overrides), stack, _make_source(method)
    )
    return {
        "method": method["name"],
        "family": family,
        "seed": seed,
        "driving_score": res.driving_score,
        "route_completion": res.route_completion,
        "complete": float(res.route_completion >= 1.0),
        "failed": float(ood_failed(res, family)),
        "collisions": sum(v for k, v in res.infractions.items() if k.startswith("collision_")),
        "max_lat_accel_ratio": res.max_lat_accel_ratio,
        "slack_friction_steps": res.slack_friction_steps,
        "deadlock": int(res.deadlock),
        "duration": res.duration,
    }


def _bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def _summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for method, family in dict.fromkeys((r["method"], r["family"]) for r in rows):
        sel = [r for r in rows if r["method"] == method and r["family"] == family]
        cell = {"n": len(sel), "collisions_total": int(sum(r["collisions"] for r in sel))}
        for key, col in _METRICS.items():
            x = np.array([r[col] for r in sel], float)
            cell[key] = {"mean": float(x.mean()), "std": float(x.std()), "ci95": list(_bootstrap_ci(x))}
        out.setdefault(method, {})[family] = cell
    return out


def _fmt(m: dict, nd: int) -> str:
    return f"{m['mean']:.{nd}f} [{m['ci95'][0]:.{nd}f}, {m['ci95'][1]:.{nd}f}]"


def _report_md(summary: dict) -> str:
    lines = [
        "| method | family | n | driving score | completion | failure rate | complete frac | collisions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for method, fams in summary.items():
        for family, c in fams.items():
            lines.append(
                f"| {method} | {family} | {c['n']} | {_fmt(c['driving_score'], 1)} "
                f"| {_fmt(c['route_completion'], 3)} | {_fmt(c['failure_rate'], 3)} "
                f"| {_fmt(c['complete_frac'], 3)} | {c['collisions_total']} |"
            )
    return "\n".join(lines) + "\n"


def run_benchmark(
    cfg_path: str | Path,
    out_dir: str | Path,
    workers: int | None = None,
    policy_path: str | Path | None = None,
) -> BenchmarkReport:
    """Run every method on every scenario seed. Rows are sorted, so results do not depend on `workers`."""
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    if cfg["backend"] != "litesim":
        raise NotImplementedError(
            f"backend {cfg['backend']!r}: only litesim is implemented (CARLA arrives in G5)"
        )
    methods = [
        {**m, "path": str(policy_path)} if policy_path and m["source"] == "policy" else m
        for m in cfg["methods"]
    ]
    tasks = [
        (m, s["family"], seed, s.get("overrides"))
        for m in methods
        for s in cfg["scenarios"]
        for seed in range(s["seeds"][0], s["seeds"][0] + s["seeds"][1])
    ]
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        rows = [_run_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_run_one, tasks))
    order = {m["name"]: i for i, m in enumerate(methods)}
    rows.sort(key=lambda r: (order[r["method"]], r["family"], r["seed"]))
    summary = _summarise(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "report.md").write_text(_report_md(summary))
    return BenchmarkReport(rows, summary)
