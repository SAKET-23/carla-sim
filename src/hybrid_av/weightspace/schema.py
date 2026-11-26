"""EditPlan schema and WeightSpaceConfig (interfaces.md section 12, llm-weight-editor.md sections 1 and 5)."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    "fc1.weight": (64, 24),
    "fc1.bias": (64,),
    "fc2.weight": (64, 64),
    "fc2.bias": (64,),
    "fc3.weight": (3, 64),
    "fc3.bias": (3,),
}


class WeightEdit(BaseModel):
    tensor: Literal["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
    index: list[int]  # [row, col] for weights, [i] for biases
    op: Literal["set", "add", "scale"]
    value: float = Field(allow_inf_nan=False)
    rationale: str

    @model_validator(mode="after")
    def _check_index(self) -> "WeightEdit":
        shape = TENSOR_SHAPES[self.tensor]
        if len(self.index) != len(shape) or any(not 0 <= i < n for i, n in zip(self.index, shape)):
            raise ValueError(f"index {self.index} out of range for {self.tensor} with shape {shape}")
        return self


class EditPlan(BaseModel):
    # 1024 bounds input size only; the max_edits limit is a gate on distinct changed parameters
    edits: list[WeightEdit] = Field(min_length=1, max_length=1024)
    hypothesis: str
    expected_effect: str


@dataclass(frozen=True, slots=True)
class WeightSpaceConfig:
    max_edits: int = 208
    max_value_factor: float = 3.0
    max_iterations: int = 5
    retention_threshold: float = 0.95
    ood_success_threshold: float = 0.8
    bundle_seeds: tuple[int, int] = (50000, 50009)
    gate_ood_seeds: tuple[int, int] = (50010, 50049)
    retention_seeds: tuple[int, int] = (50000, 50049)
    rollout_nominal_seeds: tuple[int, int] = (0, 9)
    rollout_budget: int = 40
    gradient_tools: bool = True
    gate_short_circuit: bool = False
    editor: str = "anthropic"
    editor_client: str = "sdk"  # sdk | claude_code (Claude Code CLI on the subscription)
    model: str = "claude-opus-5-5"
    max_tokens_per_call: int = 16000
    max_tool_calls_per_iteration: int = 60
    max_input_tokens_total: int = 3_000_000
    compact_after_input_tokens: int = 500_000  # 0 disables transcript compaction
    submit_reserve_calls: int = (
        5  # forced submit when this many tool calls (or one compaction's tokens) remain
    )
    notes_path: str | None = None  # seeds the editor notes from a previous run's notes.md
    backend: str = "litesim"
