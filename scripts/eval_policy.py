"""Run a benchmark YAML (methods x scenarios x seeds) and write rows.csv, summary.json and report.md to a run directory."""

import argparse
import os
from pathlib import Path

import yaml

from hybrid_av.config import make_run_dir
from hybrid_av.eval.benchmark import run_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--policy", type=Path, default=None, help="overrides the path of every `policy` method")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=Path("runs"))
    a = ap.parse_args()
    d = make_run_dir(
        "eval_policy",
        {"bench": yaml.safe_load(a.bench.read_text()), "args": {"policy": a.policy, "workers": a.workers}},
        root=a.out,
    )
    rep = run_benchmark(a.bench, d, workers=a.workers, policy_path=a.policy)
    print((d / "report.md").read_text())
    print(f"{len(rep.rows)} episodes -> {d}")


if __name__ == "__main__":
    main()
