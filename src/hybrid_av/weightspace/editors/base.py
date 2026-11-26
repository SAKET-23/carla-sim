"""Editor protocol shared by all editors (interfaces.md section 12)."""

from dataclasses import dataclass
from typing import Protocol

from hybrid_av.weightspace.bundle import Bundle
from hybrid_av.weightspace.schema import EditPlan
from hybrid_av.weightspace.validate import GateResult


@dataclass
class Attempt:
    iteration: int
    plan: EditPlan
    gates: list[GateResult]
    eval_summary: dict


class Editor(Protocol):
    name: str
    usage: dict[str, int]  # {"input_tokens", "output_tokens"}, cumulative; zeros for non-API editors
    last_messages: list[dict]  # last API conversation; empty for non-API editors

    def propose(self, bundle: Bundle, history: list[Attempt]) -> EditPlan: ...
