"""Run the gated weight-edit loop on a bundle with scripted, manual, random_sparse or anthropic editors."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from hybrid_av.config import load_dataclass
from hybrid_av.weightspace.editors import make_editor
from hybrid_av.weightspace.loop import run_edit_loop
from hybrid_av.weightspace.schema import EditPlan, WeightSpaceConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--editor", required=True, choices=["scripted", "manual", "random_sparse", "anthropic"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/weightspace/default.yaml"))
    ap.add_argument("--plan", type=Path, help="manual editor: plan.json")
    ap.add_argument(
        "--reference-plan", type=Path, help="random_sparse: plan whose per-tensor counts it matches"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=1, help="random_sparse only: independent one-proposal loops")
    a = ap.parse_args()
    cfg = load_dataclass(WeightSpaceConfig, a.config)

    if a.editor == "random_sparse":
        if not a.reference_plan:
            ap.error("--reference-plan is required for random_sparse")
        reference = EditPlan.model_validate_json(a.reference_plan.read_text())
        one = replace(cfg, max_iterations=1)  # each draw is one proposal through the gates
        results = [
            run_edit_loop(
                a.bundle,
                make_editor("random_sparse", cfg, reference_plan=reference, seed=a.seed + k),
                one,
                a.out / f"draw_{k}",
            )
            for k in range(a.draws)
        ]
        n = sum(r.accepted for r in results)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "draws_summary.json").write_text(json.dumps({"draws": a.draws, "accepted": n}, indent=1))
        print(f"{n}/{a.draws} draws accepted -> {a.out}")
        return

    kw = {"manual": {"plan_path": a.plan}}.get(a.editor, {})
    if a.editor == "manual" and not a.plan:
        ap.error("--plan is required for manual")
    result = run_edit_loop(a.bundle, make_editor(a.editor, cfg, **kw), cfg, a.out)
    print(f"accepted={result.accepted} attempts={len(result.attempts)} -> {a.out}")


if __name__ == "__main__":
    main()
