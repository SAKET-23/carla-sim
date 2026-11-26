"""Failure bundle export and load (llm-weight-editor.md section 2)."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hybrid_av.eval.metrics import ood_failed
from hybrid_av.eval.runner import EpisodeResult
from hybrid_av.features.spec import FEATURES, denormalize
from hybrid_av.policy.io import load_policy, policy_hash, policy_to_json
from hybrid_av.policy.network import PARAM_COUNT, CompactPolicy
from hybrid_av.weightspace.schema import TENSOR_SHAPES

_G = 9.81
_COLLISION_CODES = (1, 2, 3)  # INFRACTION_KEYS index + 1 of collision_*
_OFF_ROAD_CODE = 6
_WINDOW = 10

_CRITERIA = {
    "wet_curve": "Failure if any of: a collision; off-road; lateral acceleration ratio |vx*r|/(mu_true*9.81) > 1.0 "
    "(slid past the friction limit); more than 10 steps with the MPC friction slack active.",
    "merge_deadlock": "Failure if any of: a collision; the vehicle is blocked (deadlock); route completion < 1 at the "
    "time limit.",
}
_OUTPUTS = [
    {"name": "alpha_v", "scaling": "0.6*(y0+1)", "range": [0.0, 1.2], "meaning": "speed-reference scale"},
    {
        "name": "delta_y",
        "scaling": "3.5*y1",
        "range": [-3.5, 3.5],
        "meaning": "lateral offset of the reference, m",
    },
    {
        "name": "brake",
        "scaling": "0.5*(y2+1)",
        "range": [0.0, 1.0],
        "meaning": "brake flag; > 0.5 requests braking",
    },
]


def _first(mask: np.ndarray, nth: int = 0) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[nth]) if len(idx) > nth else len(mask) - 1


def classify_failure(result: EpisodeResult, family: str) -> tuple[bool, str, int]:
    """(failed, reason, failure tick index) per scenarios.md sections 2-3; first matching reason wins."""
    inf, tr = result.infractions, result.trace
    collided = any(v > 0 for k, v in inf.items() if k.startswith("collision_"))
    collision_tick = _first(np.isin(tr["infraction"], _COLLISION_CODES))
    last = len(tr["t"]) - 1
    if family == "wet_curve":
        ratio = np.abs(tr["vx"] * tr["r"]) / (tr["mu_true"] * _G)
        slack = tr["slack_f"] > 1e-3
        checks = [
            (collided, "collision", collision_tick),
            (inf.get("off_road_m", 0) > 0, "off_road", _first(tr["infraction"] == _OFF_ROAD_CODE)),
            (bool(np.any(ratio > 1.0)), "lat_accel_ratio", _first(ratio > 1.0)),
            (int(slack.sum()) > 10, "friction_slack", _first(slack, 10)),
        ]
    elif family == "merge_deadlock":
        checks = [
            (collided, "collision", collision_tick),
            (result.deadlock or inf.get("blocked", 0) > 0, "blocked", last),
            (result.route_completion < 1, "incomplete", last),
        ]
    else:
        raise ValueError(f"unknown scenario family {family!r}")
    for hit, reason, tick in checks:
        if hit:
            return True, reason, tick
    return False, "", -1


def _round7(x):
    return float(f"{x:.7g}") if np.isscalar(x) else [_round7(v) for v in x]


def _npz(policy: CompactPolicy, result: EpisodeResult, outcome: str, tick: int, path: Path) -> None:
    tr = result.trace
    m = tr["policy_step"].astype(bool)
    z = tr["z"][m].astype(np.float32)
    if all(k in tr for k in ("h1", "h2", "y_pre")):
        h1, h2, y_pre = tr["h1"][m], tr["h2"][m], tr["y_pre"][m]
    else:
        with torch.no_grad():
            _, a = policy.forward_with_activations(torch.as_tensor(z))
        h1, h2, y_pre = a["h1"].numpy(), a["h2"].numpy(), a["out_pre"].numpy()
    ratio = np.abs(tr["vx"] * tr["r"]) / (tr["mu_true"] * _G)
    step = int(m[: tick + 1].sum()) - 1 if tick >= 0 else -1  # last policy step at or before the failure tick
    np.savez(
        path,
        z=z,
        raw=denormalize(z),
        h1=h1,
        h2=h2,
        out_pre=y_pre,
        action=np.stack([tr["alpha_v"], tr["delta_y"], tr["brake_flag"]], axis=1)[m],
        route_s=tr["route_s"][m],
        vx=tr["vx"][m],
        lat_accel_ratio=ratio[m],
        mpc_status=tr["mpc_status"][m],
        slack_friction=tr["slack_f"][m],
        slack_obstacle=tr["slack_o"][m],
        t=tr["t"][m],
        outcome=outcome,
        failure_step=step,
    )


def _row(result: EpisodeResult, failed: bool, reason: str, kind: str) -> dict:
    return {
        "kind": kind,
        "seed": result.seed,
        "failed": failed,
        "reason": reason,
        "driving_score": result.driving_score,
        "route_completion": result.route_completion,
        "max_lat_accel_ratio": result.max_lat_accel_ratio,
        "mpc_fallbacks": result.mpc_fallbacks,
        "slack_friction_steps": result.slack_friction_steps,
        "min_ttc": result.min_ttc,
    }


def _readme(scenario: str, episodes: list[dict], n_nom: int) -> str:
    reasons = sorted({e["reason"] for e in episodes if e["kind"] == "failure"})
    names = ", ".join(f.name for f in FEATURES)
    return f"""# Weight-edit task: {scenario}

You edit the weights of a small driving-policy network by hand to fix a specific out-of-distribution
failure. You may not train it.

## Network
24-64-64-3 MLP (relu, relu, tanh), 5,955 parameters; tensors fc1.weight (64,24), fc1.bias (64),
fc2.weight (64,64), fc2.bias (64), fc3.weight (3,64), fc3.bias (3). See architecture.json, feature_spec.json
(inputs: {names}) and output_spec.json (alpha_v, delta_y, brake and their effect through the reference
generator and MPC).

## The failure
{_CRITERIA[scenario]} This bundle holds {sum(e["kind"] == "failure" for e in episodes)} failing episodes
(reasons: {", ".join(reasons)}) in failures/ and {n_nom} nominal reference episodes in nominal/.
Per-step traces (z, activations, actions, MPC status) are in the .npz files; outcomes are in summary.json.

## Constraints
At most 208 changed parameters; every new value within 3.0 x max(1.0, max |w| of that tensor in the base
policy); the edited policy must keep nominal driving (retention >= 0.95) and fix held-out seeds you cannot see.

## Method (guidance, not a recipe)
Compare activations on failure and nominal states, find units that are dead or inactive on nominal data, and
check an edit's output shift offline before using rollouts.

## Output
One EditPlan JSON (hypothesis, expected_effect, edits[tensor, index, op set|add|scale, value, rationale]),
submitted once you are confident.
"""


def export_bundle(
    policy: CompactPolicy,
    failures: list[EpisodeResult],
    nominal_ref: list[EpisodeResult],
    out_dir: Path,
    policy_path: Path,
) -> Path:
    scenarios = {r.scenario for r in failures}
    if len(scenarios) != 1:
        raise ValueError(f"bundle needs failures from exactly one scenario, got {sorted(scenarios)}")
    scenario = scenarios.pop()
    phash = policy_hash(policy)
    root = Path(out_dir) / f"bundle_{scenario}_{phash[:8]}"
    (root / "failures").mkdir(parents=True, exist_ok=True)
    (root / "nominal").mkdir(exist_ok=True)

    episodes = []
    for r in failures:
        _, reason, tick = classify_failure(r, scenario)
        _npz(policy, r, reason, tick, root / "failures" / f"seed_{r.seed}.npz")
        episodes.append(_row(r, True, reason, "failure"))
    for r in nominal_ref:
        failed = ood_failed(r, "nominal")
        _npz(policy, r, "failed" if failed else "ok", -1, root / "nominal" / f"seed_{r.seed}.npz")
        episodes.append(_row(r, failed, "", "nominal"))

    shutil.copy(policy_path, root / "policy.pt")
    # policy-mlp.md section 6: JSON floats at 7 significant digits (reload reproduces outputs within 1e-6)
    wj = {k: _round7(np.asarray(v)) for k, v in policy_to_json(policy).items()}
    files = {
        "policy.json": wj,
        "feature_spec.json": [
            {
                "index": f.index,
                "name": f.name,
                "unit": f.unit,
                "lo": f.lo,
                "hi": f.hi,
                "description": f.source_gt,
            }
            for f in FEATURES
        ],
        "output_spec.json": _OUTPUTS,
        "architecture.json": {
            "layers": [{"name": k, "shape": list(s)} for k, s in TENSOR_SHAPES.items()],
            "activations": ["relu", "relu", "tanh"],
            "param_count": PARAM_COUNT,
            "state_dict_keys": list(TENSOR_SHAPES),
        },
        "scenario.json": {"scenario": scenario, "failure_criteria": _CRITERIA[scenario]},
        "summary.json": {"scenario": scenario, "policy_hash": phash, "episodes": episodes},
    }
    for name, data in files.items():
        (root / name).write_text(json.dumps(data, indent=1))
    (root / "README.md").write_text(_readme(scenario, episodes, len(nominal_ref)))
    return root


@dataclass
class Bundle:
    root: Path
    scenario: str
    policy: CompactPolicy
    policy_hash: str
    feature_spec: list[dict]
    summary: list[dict]
    traces: dict[
        int, dict[str, np.ndarray]
    ]  # failure and nominal traces by seed (the seed ranges are disjoint)
    Z_fail: np.ndarray
    Z_window: np.ndarray
    Z_nom: np.ndarray


def load_bundle(path: Path) -> Bundle:
    root = Path(path)
    summary = json.loads((root / "summary.json").read_text())
    policy, _ = load_policy(root / "policy.pt")
    actual = policy_hash(policy)
    if actual != summary["policy_hash"]:
        raise ValueError(
            f"policy hash mismatch: policy.pt {actual[:8]} vs summary.json {summary['policy_hash'][:8]}"
        )

    def load(sub: str) -> dict[int, dict[str, np.ndarray]]:
        return {int(f.stem.split("_")[1]): dict(np.load(f)) for f in sorted((root / sub).glob("seed_*.npz"))}

    fail, nom = load("failures"), load("nominal")
    empty = np.zeros((0, 24), np.float32)
    windows = [
        t["z"][max(0, int(t["failure_step"]) - _WINDOW) : int(t["failure_step"])] for t in fail.values()
    ]
    return Bundle(
        root=root,
        scenario=summary["scenario"],
        policy=policy,
        policy_hash=actual,
        feature_spec=json.loads((root / "feature_spec.json").read_text()),
        summary=summary["episodes"],
        traces={**fail, **nom},
        Z_fail=np.concatenate([t["z"] for t in fail.values()]) if fail else empty,
        Z_window=np.concatenate(windows) if windows else empty,
        Z_nom=np.concatenate([t["z"] for t in nom.values()]) if nom else empty,
    )
