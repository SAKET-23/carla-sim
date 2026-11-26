"""Gated edit loop with per-iteration logs (llm-weight-editor.md sections 7 and 8, interfaces.md section 12)."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hybrid_av.policy.io import policy_hash, save_policy
from hybrid_av.policy.network import CompactPolicy
from hybrid_av.weightspace.apply import EditDiff, apply_edit_plan
from hybrid_av.weightspace.bundle import load_bundle
from hybrid_av.weightspace.editors.anthropic_editor import EditorGaveUp
from hybrid_av.weightspace.editors.base import Attempt, Editor
from hybrid_av.weightspace.schema import WeightSpaceConfig
from hybrid_av.weightspace.validate import (
    EvalSummary,
    behavioural_gates,
    default_stack_factory,
    make_eval_fn,
    static_gates,
)


@dataclass
class EditLoopResult:
    accepted: bool
    final_policy_path: Path | None
    attempts: list[Attempt]
    diff: EditDiff | None


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))


def run_edit_loop(
    bundle_path: Path,
    editor: Editor,
    cfg: WeightSpaceConfig,
    out_dir: Path,
    eval_fn: Callable[[CompactPolicy], EvalSummary] | None = None,
) -> EditLoopResult:
    bundle = load_bundle(bundle_path)
    base, base_hash = bundle.policy, bundle.policy_hash
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_eval = eval_fn or make_eval_fn(bundle.scenario, cfg, default_stack_factory)
    evaluated: dict[str, EvalSummary] = {}  # the base policy is evaluated once per loop

    def cached_eval(p: CompactPolicy) -> EvalSummary:
        key = policy_hash(p)
        if key not in evaluated:
            evaluated[key] = raw_eval(p)
        return evaluated[key]

    states = np.concatenate([bundle.Z_fail, bundle.Z_nom])
    attempts: list[Attempt] = []
    accepted, stop, diff, final_hash, final_path = False, "max_iterations", None, base_hash, None
    prev_usage = dict(editor.usage)
    if hasattr(editor, "notes_dir"):
        editor.notes_dir = out

    def log(i: int, plan, gates, summary) -> None:
        nonlocal prev_usage
        d = out / "attempts" / f"iter_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "messages.jsonl").write_text(
            "".join(json.dumps(m, default=str) + "\n" for m in editor.last_messages)
        )
        if plan is not None:
            _write(d / "plan.json", plan.model_dump())
        _write(d / "gates.json", [asdict(g) for g in gates])
        _write(d / "eval_summary.json", summary)
        now = dict(editor.usage)
        _write(
            d / "usage.json",
            {**{k: now[k] - prev_usage[k] for k in now}, **{f"cumulative_{k}": v for k, v in now.items()}},
        )
        prev_usage = now

    for i in range(1, cfg.max_iterations + 1):
        if editor.usage["input_tokens"] >= cfg.max_input_tokens_total:
            stop = "token_cap"
            break
        try:
            plan = editor.propose(bundle, list(attempts))
        except EditorGaveUp:
            stop = (
                "token_cap"
                if editor.usage["input_tokens"] >= cfg.max_input_tokens_total
                else "editor_gave_up"
            )
            log(i, None, [], {})
            break
        gates = static_gates(plan, base, cfg, states)
        if not all(g.passed for g in gates):
            attempts.append(Attempt(i, plan, gates, {}))
            log(i, plan, gates, {})
            continue
        edited, edit_diff = apply_edit_plan(base, plan)
        behav = behavioural_gates(edited, base, cached_eval, cfg)
        if cfg.gate_short_circuit and not behav[0].passed:
            behav = behav[:1]  # rejected by the OOD gate; retention is not reported
        gates += behav
        summary = cached_eval(edited).to_feedback()
        attempts.append(Attempt(i, plan, gates, summary))
        log(i, plan, gates, summary)
        if all(g.passed for g in gates):
            accepted, stop, diff, final_hash = True, "accepted", edit_diff, policy_hash(edited)
            final_path = out / "policy_edited.pt"
            save_policy(edited, final_path, {"created_by": f"edit:{editor.name}", "parent_hash": base_hash})
            _write(out / "edit_plan.json", plan.model_dump())
            break

    _write(
        out / "summary.json",
        {
            "accepted": accepted,
            "stop_reason": stop,
            "iterations": len(attempts),
            "usage": dict(editor.usage),
            "compactions": getattr(editor, "compactions", 0),
            "env_steps": {
                "gates": sum(s.env_steps for s in evaluated.values()),
                "rollouts": getattr(editor, "rollout_steps", 0),
            },
            "diff": asdict(diff) if diff else None,
            "final_policy_hash": final_hash,
            "base_policy_hash": base_hash,
        },
    )
    return EditLoopResult(accepted, final_path, attempts, diff)
