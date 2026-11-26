"""Random-sparse control (llm-weight-editor.md section 9, weight-space-editing.md section 7)."""

import numpy as np

from hybrid_av.weightspace.apply import apply_edit_plan
from hybrid_av.weightspace.schema import TENSOR_SHAPES, EditPlan, WeightEdit


class RandomSparseEditor:
    name = "random_sparse"

    def __init__(self, reference_plan: EditPlan, seed: int) -> None:
        self.reference_plan, self.seed, self._k = reference_plan, seed, 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.last_messages: list[dict] = []

    def propose(self, bundle, history) -> EditPlan:
        base = bundle.policy
        edited, diff = apply_edit_plan(base, self.reference_plan)
        # realised |delta| of every changed parameter of the reference plan
        mags = np.concatenate(
            [
                (edited.state_dict()[k].double() - base.state_dict()[k].double()).abs().flatten().numpy()
                for k in diff.per_tensor
            ]
        )
        mags = mags[mags > 0]
        rng = np.random.default_rng(self.seed + self._k)
        self._k += 1
        edits = []
        for tensor, n in diff.per_tensor.items():
            shape = TENSOR_SHAPES[tensor]
            for flat in rng.choice(int(np.prod(shape)), size=n, replace=False):
                value = float(rng.choice(mags) * rng.choice([-1.0, 1.0]))
                edits.append(
                    WeightEdit(
                        tensor=tensor,
                        index=[int(i) for i in np.unravel_index(flat, shape)],
                        op="add",
                        value=value,
                        rationale="random sparse control",
                    )
                )
        return EditPlan(
            edits=edits,
            hypothesis="Random-sparse control: no hypothesis.",
            expected_effect="None expected; matches the reference plan's per-tensor count and |delta| distribution.",
        )
