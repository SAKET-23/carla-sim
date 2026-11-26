"""Manual editor: returns the plan stored in a JSON file."""

from pathlib import Path

from hybrid_av.weightspace.schema import EditPlan


class ManualEditor:
    name = "manual"

    def __init__(self, plan_path: Path) -> None:
        self.plan_path = Path(plan_path)
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.last_messages: list[dict] = []

    def propose(self, bundle, history) -> EditPlan:
        return EditPlan.model_validate_json(self.plan_path.read_text())
