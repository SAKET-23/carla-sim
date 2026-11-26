# ruff: noqa
import json, numpy as np
from dataclasses import replace
from hybrid_av.weightspace.loop import run_edit_loop
from hybrid_av.weightspace.schema import EditPlan, WeightEdit, WeightSpaceConfig
from hybrid_av.weightspace.validate import EvalSummary

CFG = replace(WeightSpaceConfig(), max_input_tokens_total=15)


def e(t, i, v):
    return WeightEdit(tensor=t, index=i, op="set", value=v, rationale="r")


NOOP = EditPlan(edits=[e("fc1.weight", [0, 7], 1.0)], hypothesis="h", expected_effect="e")  # fails "count"
WEAK = EditPlan(edits=[e("fc1.bias", [1], 0.5)], hypothesis="h", expected_effect="e")
GOOD = EditPlan(edits=[e("fc1.bias", [2], 0.5)], hypothesis="h", expected_effect="e")


class SeqEditor:
    name = "seq"

    def __init__(self, plans, tokens=0):
        self.plans, self.tokens, self.usage, self.last_messages, self.histories = (
            list(plans),
            tokens,
            {"input_tokens": 0, "output_tokens": 0},
            [],
            [],
        )

    def propose(self, bundle, history):
        self.histories.append(history)
        self.usage["input_tokens"] += self.tokens
        return self.plans.pop(0)


def eval_fn(p):
    fixed = float(p.fc1.bias[2]) == 0.5
    return EvalSummary(
        0.9 if fixed else 0.1,
        {} if fixed else {"lat_accel_ratio": 36},
        np.array([90.0, 90.0]),
        frozenset({50004}),
        0,
    )


def test_loop_accepts_third_plan(tmp_path, bundle_path):
    ed = SeqEditor([NOOP, WEAK, GOOD])
    r = run_edit_loop(bundle_path, ed, CFG, tmp_path, eval_fn=eval_fn)
    assert r.accepted and len(r.attempts) == 3 and [len(h) for h in ed.histories] == [0, 1, 2]
    assert r.attempts[0].gates[0].gate == "count" and not r.attempts[0].gates[0].passed
    assert r.diff.n_changed == 1
    for i in (1, 2, 3):
        for f in ("plan.json", "gates.json", "eval_summary.json", "usage.json", "messages.jsonl"):
            assert (tmp_path / f"attempts/iter_{i}/{f}").exists()
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["accepted"] and s["stop_reason"] == "accepted"
    assert (tmp_path / "policy_edited.pt").exists() and (tmp_path / "edit_plan.json").exists()


def test_history_has_no_per_seed_gate_traces(tmp_path, bundle_path):
    ed = SeqEditor([WEAK, GOOD])
    run_edit_loop(bundle_path, ed, CFG, tmp_path, eval_fn=eval_fn)
    assert "50004" not in str(ed.histories[1][0])  # neither eval_summary nor gate details carry seed ids


def test_loop_stops_at_token_cap(tmp_path, bundle_path):
    r = run_edit_loop(bundle_path, SeqEditor([WEAK] * 5, tokens=10), CFG, tmp_path, eval_fn=eval_fn)
    assert not r.accepted and len(r.attempts) == 2 and r.final_policy_path is None
    assert json.loads((tmp_path / "summary.json").read_text())["stop_reason"] == "token_cap"


def test_loop_stops_at_max_iterations(tmp_path, bundle_path):
    r = run_edit_loop(bundle_path, SeqEditor([WEAK] * 5), CFG, tmp_path, eval_fn=eval_fn)
    assert not r.accepted and len(r.attempts) == 5


def test_short_circuit_reports_only_ood_gate_and_gave_up_stop(tmp_path, bundle_path):
    from hybrid_av.weightspace.editors.anthropic_editor import EditorGaveUp

    cfg = replace(WeightSpaceConfig(), gate_short_circuit=True, max_iterations=1)
    r = run_edit_loop(bundle_path, SeqEditor([WEAK]), cfg, tmp_path / "a", eval_fn=eval_fn)
    assert [g.gate for g in r.attempts[0].gates][-1] == "ood_success" and not r.attempts[0].gates[-1].passed

    class Quits(SeqEditor):
        def propose(self, bundle, history):
            raise EditorGaveUp("no")

    r = run_edit_loop(bundle_path, Quits([]), CFG, tmp_path / "b", eval_fn=eval_fn)
    assert json.loads((tmp_path / "b/summary.json").read_text())["stop_reason"] == "editor_gave_up"
