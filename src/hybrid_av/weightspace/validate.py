"""Static and behavioural acceptance gates (llm-weight-editor.md section 6, interfaces.md section 12).

Gate details reach the editor through the history: behavioural details carry counts and thresholds only, never seeds.
"""

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from hybrid_av.eval.metrics import ood_failed
from hybrid_av.eval.runner import run_episode
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.policy.io import policy_hash
from hybrid_av.policy.network import CompactPolicy, TorchPolicySource
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from hybrid_av.weightspace.apply import apply_edit_plan
from hybrid_av.weightspace.bundle import classify_failure
from hybrid_av.weightspace.schema import TENSOR_SHAPES, EditPlan, WeightSpaceConfig


@dataclass
class GateResult:
    passed: bool
    gate: str
    detail: str


@dataclass
class EvalSummary:
    ood_success_rate: float
    ood_failure_reasons: dict[str, int]
    nominal_scores: np.ndarray
    nominal_collision_seeds: frozenset[int]
    nominal_mpc_fallbacks: int
    env_steps: int = 0

    def to_feedback(self) -> dict:
        return {
            "ood_success_rate": self.ood_success_rate,
            "ood_failure_reasons": dict(self.ood_failure_reasons),
            "nominal_mean_score": float(np.mean(self.nominal_scores)),
            "nominal_collisions": len(self.nominal_collision_seeds),
            "nominal_mpc_fallbacks": self.nominal_mpc_fallbacks,
        }


def _addr(tensor: str, index: list[int]) -> str:
    return f"{tensor}[{', '.join(map(str, index))}]"


def static_gates(
    plan: EditPlan, policy: CompactPolicy, cfg: WeightSpaceConfig, bundle_states: np.ndarray | None = None
) -> list[GateResult]:
    edited, diff = apply_edit_plan(policy, plan)
    count = GateResult(
        1 <= diff.n_changed <= cfg.max_edits,
        "count",
        f"{diff.n_changed} parameters changed (allowed 1..{cfg.max_edits})",
    )

    bad_idx = [
        _addr(e.tensor, e.index)
        for e in plan.edits
        if len(e.index) != len(TENSOR_SHAPES[e.tensor])
        or any(not 0 <= i < n for i, n in zip(e.index, TENSOR_SHAPES[e.tensor]))
    ]
    index_range = GateResult(not bad_idx, "index_range", f"out of range: {bad_idx}" if bad_idx else "ok")

    base_sd, new_sd = policy.state_dict(), edited.state_dict()
    bad_val = []
    for e in plan.edits:
        if bad_idx:
            break  # indexing below would raise; index_range already failed
        limit = cfg.max_value_factor * max(1.0, float(base_sd[e.tensor].abs().max()))
        v = float(new_sd[e.tensor][tuple(e.index)])
        if not np.isfinite(v) or abs(v) > limit:
            bad_val.append(f"{_addr(e.tensor, e.index)}={v:g} (limit {limit:g})")
    value_range = GateResult(not bad_val, "value_range", f"out of range: {bad_val}" if bad_val else "ok")

    if bundle_states is None or bad_idx:
        finite, detail = True, "no bundle states"
    else:
        with torch.no_grad():
            _, acts = edited.forward_with_activations(torch.as_tensor(bundle_states, dtype=torch.float32))
        bad = [
            k
            for k in ("h1_pre", "h2_pre", "out_pre", "out")
            if k in acts and not torch.isfinite(acts[k]).all()
        ]
        finite, detail = not bad, f"non-finite activations: {bad}" if bad else "ok"
    return [count, index_range, value_range, GateResult(finite, "finite_outputs", detail)]


def behavioural_gates(
    edited: CompactPolicy,
    base: CompactPolicy,
    eval_fn: Callable[[CompactPolicy], EvalSummary],
    cfg: WeightSpaceConfig,
) -> list[GateResult]:
    e, b = eval_fn(edited), eval_fn(base)
    ret = float(np.mean(e.nominal_scores)) / float(np.mean(b.nominal_scores))
    new_coll = len(e.nominal_collision_seeds - b.nominal_collision_seeds)
    return [
        GateResult(
            e.ood_success_rate >= cfg.ood_success_threshold,
            "ood_success",
            f"OOD success {e.ood_success_rate:.3f} (threshold {cfg.ood_success_threshold})",
        ),
        GateResult(
            ret >= cfg.retention_threshold,
            "retention",
            f"retention {ret:.3f} (threshold {cfg.retention_threshold})",
        ),
        GateResult(not new_coll, "no_new_collisions", f"{new_coll} new nominal collisions (allowed 0)"),
        GateResult(
            e.nominal_mpc_fallbacks <= b.nominal_mpc_fallbacks + 1,
            "mpc_fallbacks",
            f"{e.nominal_mpc_fallbacks} MPC fallbacks vs base {b.nominal_mpc_fallbacks} (allowed base + 1)",
        ),
    ]


def _run_gate_episode(job: tuple) -> tuple[int, object]:
    """Module-level so spawned workers can import it; the policy travels as its state_dict."""
    stack_factory, state_dict, family, seed = job
    policy = CompactPolicy()
    policy.load_state_dict(state_dict)
    stack = stack_factory()
    sim = LiteSim(stack.params, stack.lm)
    return seed, run_episode(sim, make_scenario(family, seed), stack, TorchPolicySource(policy))


def default_stack_factory() -> AutonomyStack:
    return AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig())


def make_eval_fn(
    family: str,
    cfg: WeightSpaceConfig,
    stack_factory: Callable[[], AutonomyStack],
    workers: int | None = None,
) -> Callable[[CompactPolicy], EvalSummary]:
    """Gate evaluation on cfg.backend with the Phase 3 run_episode, cached by policy hash.

    Episodes run in `workers` processes (default os.cpu_count()); results are ordered by task, so
    they do not depend on the worker count.
    """
    if cfg.backend != "litesim":
        raise NotImplementedError(
            f"backend {cfg.backend!r}: only litesim is implemented (CARLA arrives in G5)"
        )
    cache: dict[str, EvalSummary] = {}
    workers = workers or os.cpu_count() or 1

    def run(policy: CompactPolicy, tasks: list[tuple[str, int]]) -> list[tuple[int, object]]:
        """Episodes for (family, seed) tasks, in task order whatever `workers` is."""
        jobs = [(stack_factory, policy.state_dict(), fam, seed) for fam, seed in tasks]
        if workers == 1:
            return [_run_gate_episode(j) for j in jobs]
        with ProcessPoolExecutor(workers) as ex:  # stack_factory must be picklable here
            return list(ex.map(_run_gate_episode, jobs))

    def eval_fn(policy: CompactPolicy) -> EvalSummary:
        key = policy_hash(policy)
        if key not in cache:
            ood_seeds, nom_seeds = (
                range(cfg.gate_ood_seeds[0], cfg.gate_ood_seeds[1] + 1),
                range(cfg.retention_seeds[0], cfg.retention_seeds[1] + 1),
            )
            res = run(policy, [(family, s) for s in ood_seeds] + [("nominal", s) for s in nom_seeds])
            ood, nom = res[: len(ood_seeds)], res[len(ood_seeds) :]
            reasons: dict[str, int] = {}
            for _, r in ood:
                if ood_failed(r, family):
                    reason = classify_failure(r, family)[1]
                    reasons[reason] = reasons.get(reason, 0) + 1
            cache[key] = EvalSummary(
                ood_success_rate=1 - sum(reasons.values()) / len(ood),
                ood_failure_reasons=reasons,
                nominal_scores=np.array([r.driving_score for _, r in nom]),
                nominal_collision_seeds=frozenset(
                    s
                    for s, r in nom
                    if any(v > 0 for k, v in r.infractions.items() if k.startswith("collision_"))
                ),
                nominal_mpc_fallbacks=sum(r.mpc_fallbacks for _, r in nom),
                env_steps=sum(len(r.trace["t"]) for _, r in ood + nom),
            )
        return cache[key]

    return eval_fn
