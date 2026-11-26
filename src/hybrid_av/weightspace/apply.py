"""Apply an EditPlan to a copy of a policy and measure the resulting diff (interfaces.md section 12)."""

import copy
from dataclasses import dataclass

import torch

from hybrid_av.policy.network import PARAM_COUNT, CompactPolicy
from hybrid_av.weightspace.schema import EditPlan


@dataclass
class EditDiff:
    n_changed: int
    fraction_changed: float
    l2_delta: float
    max_abs_delta: float
    per_tensor: dict[str, int]


def diff_policies(a: CompactPolicy, b: CompactPolicy, tol: float = 0.0) -> EditDiff:
    per_tensor: dict[str, int] = {}
    sq, mx = 0.0, 0.0
    sa, sb = a.state_dict(), b.state_dict()
    for k in sa:
        d = (sb[k].double() - sa[k].double()).abs()
        changed = d > tol
        if changed.any():
            per_tensor[k] = int(changed.sum())
            sq += float((d[changed] ** 2).sum())
            mx = max(mx, float(d[changed].max()))
    n = sum(per_tensor.values())
    return EditDiff(n, n / PARAM_COUNT, sq**0.5, mx, per_tensor)


def apply_edit_plan(policy: CompactPolicy, plan: EditPlan) -> tuple[CompactPolicy, EditDiff]:
    edited = copy.deepcopy(policy)
    params = dict(edited.named_parameters())
    with torch.no_grad():
        for e in plan.edits:
            t = params[e.tensor]
            i = tuple(e.index)
            if e.op == "set":
                t[i] = e.value
            elif e.op == "add":
                t[i] += e.value
            else:
                t[i] *= e.value
    return edited, diff_policies(policy, edited)
