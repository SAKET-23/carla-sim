"""Export a failure bundle for a policy: OOD seeds of one scenario plus 5 nominal reference episodes."""

import argparse
import sys
from pathlib import Path

from hybrid_av.config import load_dataclass
from hybrid_av.eval.runner import run_episode
from hybrid_av.policy.io import load_policy
from hybrid_av.policy.network import TorchPolicySource
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.weightspace.bundle import classify_failure, export_bundle
from hybrid_av.weightspace.schema import WeightSpaceConfig
from hybrid_av.weightspace.validate import default_stack_factory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--scenario", required=True, choices=["wet_curve", "merge_deadlock"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/weightspace/default.yaml"))
    a = ap.parse_args()
    cfg = load_dataclass(WeightSpaceConfig, a.config)
    policy, _ = load_policy(a.policy)

    def run(family: str, seed: int):
        stack = default_stack_factory()
        sim = LiteSim(stack.params, stack.lm)
        return run_episode(sim, make_scenario(family, seed), stack, TorchPolicySource(policy))

    # activations are recomputed from the stored states by export_bundle (run_episode's record_activations is unused)
    ood = [run(a.scenario, s) for s in range(cfg.bundle_seeds[0], cfg.bundle_seeds[1] + 1)]
    failures = [r for r in ood if classify_failure(r, a.scenario)[0]]
    if not failures:
        print(
            f"no {a.scenario} failures on bundle seeds {cfg.bundle_seeds}; nothing to export", file=sys.stderr
        )
        return 2
    lo = cfg.rollout_nominal_seeds[0]
    nominal = [run("nominal", s) for s in range(lo, lo + 5)]
    root = export_bundle(policy, failures, nominal, a.out, a.policy)
    print(f"{len(failures)}/{len(ood)} failing episodes -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
