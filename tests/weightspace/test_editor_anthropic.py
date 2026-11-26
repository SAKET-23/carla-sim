# ruff: noqa
import sys, pytest
from dataclasses import replace
from tests.fakes.anthropic_fake import FakeAnthropic, tool_use, text, response
import dataclasses
from hybrid_av.weightspace.editors.anthropic_editor import AnthropicEditor, EditorGaveUp, TOOLS
from hybrid_av.weightspace.schema import WeightSpaceConfig

CFG = replace(WeightSpaceConfig(), rollout_budget=1, max_tool_calls_per_iteration=6, submit_reserve_calls=0)
GOOD = {
    "edits": [{"tensor": "fc3.bias", "index": [0], "op": "add", "value": -0.3, "rationale": "r"}],
    "hypothesis": "h",
    "expected_effect": "e",
}
BAD = {**GOOD, "edits": [{**GOOD["edits"][0], "tensor": "fc3.weight", "index": [3, 0]}]}
NOROLL = lambda p, s, seed: {"failed": False}


def ed(fake, roll=NOROLL):
    return AnthropicEditor(CFG, client=fake, rollout_fn=roll)


def test_tool_names():
    assert [t["name"] for t in TOOLS] == [
        "get_tensor",
        "activation_stats",
        "neuron_feature_attribution",
        "output_attribution",
        "counterfactual",
        "evaluate_plan_offline",
        "run_rollout",
        "submit_edit_plan",
    ]


def test_gradient_tools_flag_removes_attribution_tool():
    from hybrid_av.weightspace.editors.anthropic_editor import tools_for

    names = [t["name"] for t in tools_for(dataclasses.replace(CFG, gradient_tools=False))]
    assert "neuron_feature_attribution" not in names and "submit_edit_plan" in names
    assert len(tools_for(CFG)) == len(TOOLS)


def test_happy_path_with_one_analysis_call(bundle_fixture):
    fake = FakeAnthropic(
        [
            response(
                [tool_use("t1", "activation_stats", {"state_set": "failure_window"})], "tool_use", 1000, 200
            ),
            response([tool_use("t2", "submit_edit_plan", GOOD)], "tool_use", 1500, 300),
        ]
    )
    e = ed(fake)
    plan = e.propose(bundle_fixture, [])
    assert plan.edits[0].value == -0.3 and len(fake.calls) == 2
    assert fake.calls[0]["model"] == "claude-opus-5-5" and fake.calls[0]["max_tokens"] == 16000
    assert e.usage == {"input_tokens": 2500, "output_tokens": 500}
    assert "anthropic" not in sys.modules


def test_invalid_submission_returns_tool_error(bundle_fixture):
    fake = FakeAnthropic(
        [
            response([tool_use("t1", "submit_edit_plan", BAD)], "tool_use", 10, 10),
            response([tool_use("t2", "submit_edit_plan", GOOD)], "tool_use", 10, 10),
        ]
    )
    ed(fake).propose(bundle_fixture, [])
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert (block["type"], block["tool_use_id"], block["is_error"]) == ("tool_result", "t1", True)
    assert "out of range" in block["content"]


def test_rollout_tool_rejects_gate_seed(bundle_fixture):
    fake = FakeAnthropic(
        [
            response(
                [tool_use("t1", "run_rollout", {"plan": GOOD, "scenario": "wet_curve", "seeds": [50020]})],
                "tool_use",
                1,
                1,
            ),
            response([tool_use("t2", "submit_edit_plan", GOOD)], "tool_use", 1, 1),
        ]
    )
    ed(fake).propose(bundle_fixture, [])
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True and "50000" in block["content"] and "50009" in block["content"]


def test_rollout_budget(bundle_fixture):
    call = {"plan": GOOD, "scenario": "wet_curve", "seeds": [50000]}
    fake = FakeAnthropic(
        [
            response(
                [tool_use("a", "run_rollout", call), tool_use("b", "run_rollout", call)], "tool_use", 1, 1
            ),
            response([tool_use("c", "submit_edit_plan", GOOD)], "tool_use", 1, 1),
        ]
    )
    ed(fake).propose(bundle_fixture, [])
    blocks = fake.calls[1]["messages"][-1]["content"]
    assert blocks[0].get("is_error") is not True and blocks[1]["is_error"] is True
    assert "rollout budget exhausted" in blocks[1]["content"]


def test_get_tensor_size_cap(bundle_fixture):
    fake = FakeAnthropic(
        [
            response([tool_use("a", "get_tensor", {"name": "fc2.weight"})], "tool_use", 1, 1),
            response([tool_use("b", "submit_edit_plan", GOOD)], "tool_use", 1, 1),
        ]
    )
    ed(fake).propose(bundle_fixture, [])
    assert fake.calls[1]["messages"][-1]["content"][0]["is_error"] is True  # 4,096 numbers > 2,000


def test_gives_up_on_end_turn_without_plan(bundle_fixture):
    with pytest.raises(EditorGaveUp):
        ed(FakeAnthropic([response([text("I cannot help")], "end_turn", 1, 1)])).propose(bundle_fixture, [])


def test_messages_never_contain_api_key(bundle_fixture, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-123")
    e = ed(FakeAnthropic([response([tool_use("t", "submit_edit_plan", GOOD)], "tool_use", 1, 1)]))
    e.propose(bundle_fixture, [])
    assert "sk-secret-123" not in str(e.last_messages)


def test_importing_weightspace_and_editor_module_does_not_import_anthropic():
    import subprocess

    code = (
        "import sys, hybrid_av.weightspace, hybrid_av.weightspace.editors.anthropic_editor;"
        "print('anthropic' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_token_cap_checked_before_every_call(bundle_fixture):
    cfg = replace(CFG, max_input_tokens_total=1000)
    stats = tool_use("a", "activation_stats", {"state_set": "nominal"})
    fake = FakeAnthropic([response([stats], "tool_use", 1000, 1), response([stats], "tool_use", 1, 1)])
    with pytest.raises(EditorGaveUp, match="token cap"):
        AnthropicEditor(cfg, client=fake, rollout_fn=NOROLL).propose(bundle_fixture, [])
    assert len(fake.calls) == 1


def test_tool_call_cap_and_removed_tool(bundle_fixture):
    cfg = replace(CFG, max_tool_calls_per_iteration=1, gradient_tools=False)
    attr = tool_use("a", "neuron_feature_attribution", {"state_set": "nominal", "layer": "h1"})
    fake = FakeAnthropic([response([attr], "tool_use", 1, 1), response([attr, attr], "tool_use", 1, 1)])
    with pytest.raises(EditorGaveUp):
        AnthropicEditor(cfg, client=fake, rollout_fn=NOROLL).propose(bundle_fixture, [])
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True


def test_rollout_steps_accumulate_and_analysis_tools_run(bundle_fixture):
    roll = lambda p, s, seed: {"failed": False, "steps": 7}
    calls = [
        tool_use("1", "run_rollout", {"plan": GOOD, "scenario": "wet_curve", "seeds": [50000]}),
        tool_use("2", "evaluate_plan_offline", {"plan": GOOD}),
        tool_use("3", "output_attribution", {"state_set": "failures", "output": 0}),
        tool_use("4", "neuron_feature_attribution", {"state_set": "failures", "layer": "h1"}),
        tool_use("5", "counterfactual", {"seed": 50000, "step": 0, "feature": 7, "values": [0.0, 1.0]}),
        tool_use("6", "get_tensor", {"name": "fc3.weight", "rows": [0, 1]}),
    ]
    fake = FakeAnthropic(
        [
            response(calls, "tool_use", 1, 1),
            response([tool_use("s", "submit_edit_plan", GOOD)], "tool_use", 1, 1),
        ]
    )
    e = AnthropicEditor(replace(CFG, max_tool_calls_per_iteration=10), client=fake, rollout_fn=roll)
    e.propose(bundle_fixture, [])
    blocks = fake.calls[1]["messages"][-1]["content"]
    assert [b.get("is_error") for b in blocks] == [None] * 6 and e.rollout_steps == 7


# ---- transcript compaction ----
def _stats(i):
    return tool_use(f"s{i}", "activation_stats", {"state_set": "nominal"})


def _notes(t="N1"):
    return tool_use("n", "record_notes", {"notes": t})


def _compact_fake(*extra, first=600):
    return FakeAnthropic(
        [
            response([_stats(1)], "tool_use", first, 1),  # 600 > 500 threshold
            *extra,
            response([tool_use("sub", "submit_edit_plan", GOOD)], "tool_use", 10, 1),
        ]
    )


def _cc(**kw):
    return replace(CFG, compact_after_input_tokens=500, **kw)


def test_compaction_triggers_past_threshold_and_resets_transcript(bundle_fixture, tmp_path):
    fake = _compact_fake(response([_notes("found X")], "tool_use", 700, 5))
    e = AnthropicEditor(_cc(), client=fake, rollout_fn=NOROLL)
    e.notes_dir = tmp_path
    e.propose(bundle_fixture, [])
    assert len(fake.calls) == 3
    assert [t["name"] for t in fake.calls[1]["tools"]] == ["record_notes"]
    assert len(fake.calls[1]["messages"]) == 3  # first, assistant, tool_result + ask in the last user message
    after = fake.calls[2]["messages"]
    assert len(after) == 1 and after[0]["role"] == "user" and "## Notes so far" in after[0]["content"]
    assert "found X" in after[0]["content"]
    assert e.usage["input_tokens"] == 600 + 700 + 10 and e.compactions == 1
    assert "found X" in (tmp_path / "notes.md").read_text()


def test_no_compaction_below_threshold_or_when_disabled(bundle_fixture):
    for cfg in (replace(CFG, compact_after_input_tokens=600), replace(CFG, compact_after_input_tokens=0)):
        fake = _compact_fake()
        AnthropicEditor(cfg, client=fake, rollout_fn=NOROLL).propose(bundle_fixture, [])
        assert len(fake.calls) == 2 and len(fake.calls[1]["messages"]) == 3


def test_notes_md_appends_and_later_iteration_includes_notes(bundle_fixture, tmp_path):
    fake = FakeAnthropic(
        [
            response([_stats(1)], "tool_use", 600, 1),
            response([_notes("A")], "tool_use", 1, 1),
            response([tool_use("s", "submit_edit_plan", GOOD)], "tool_use", 1, 1),
            response([_stats(2)], "tool_use", 600, 1),
            response([text("plain B")], "end_turn", 1, 1),  # no tool call: falls back to reply text
            response([tool_use("s", "submit_edit_plan", GOOD)], "tool_use", 1, 1),
        ]
    )
    e = AnthropicEditor(_cc(), client=fake, rollout_fn=NOROLL)
    e.notes_dir = tmp_path
    e.propose(bundle_fixture, [])
    e.propose(bundle_fixture, [])
    assert (
        "## Notes so far" in fake.calls[3]["messages"][0]["content"]
        and "\nA\n" in fake.calls[3]["messages"][0]["content"]
    )
    md = (tmp_path / "notes.md").read_text()
    assert (
        "compaction 1" in md
        and "compaction 2" in md
        and "plain B" in md
        and md.index("A") < md.index("plain B")
    )


def test_notes_path_seeds_first_message(bundle_fixture, tmp_path):
    f = tmp_path / "old.md"
    f.write_text("OLD FINDING")
    fake = FakeAnthropic([response([tool_use("s", "submit_edit_plan", GOOD)], "tool_use", 1, 1)])
    AnthropicEditor(replace(CFG, notes_path=str(f)), client=fake, rollout_fn=NOROLL).propose(
        bundle_fixture, []
    )
    assert "OLD FINDING" in fake.calls[0]["messages"][0]["content"]


def test_token_cap_still_enforced_after_compaction(bundle_fixture):
    fake = FakeAnthropic(
        [response([_stats(1)], "tool_use", 600, 1), response([_notes()], "tool_use", 500, 1)]
    )
    cfg = _cc(max_input_tokens_total=1000)
    with pytest.raises(EditorGaveUp, match="token cap"):
        AnthropicEditor(cfg, client=fake, rollout_fn=NOROLL).propose(bundle_fixture, [])
    assert len(fake.calls) == 2


# ---- forced submit ----
def _alternates(msgs):
    return all(a["role"] != b["role"] for a, b in zip(msgs, msgs[1:]))


def _sub(i="sub", plan=GOOD):
    return response([tool_use(i, "submit_edit_plan", plan)], "tool_use", 1, 1)


def _final_cfg(**kw):
    return replace(WeightSpaceConfig(), **{"max_tool_calls_per_iteration": 60, "rollout_budget": 1, **kw})


def test_forced_submit_triggers_at_55_of_60_and_not_earlier(bundle_fixture):
    fake = FakeAnthropic([*[response([_stats(i)], "tool_use", 1, 1) for i in range(55)], _sub()])
    plan = AnthropicEditor(_final_cfg(), client=fake, rollout_fn=NOROLL).propose(bundle_fixture, [])
    assert plan.edits[0].value == -0.3 and len(fake.calls) == 56
    assert all("tool_choice" not in c and len(c["tools"]) > 1 for c in fake.calls[:55])
    last = fake.calls[55]
    assert [t["name"] for t in last["tools"]] == ["submit_edit_plan"]
    assert last["tool_choice"] == {"type": "tool", "name": "submit_edit_plan"}
    msgs = last["messages"]
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"][-1]["text"].startswith("Budget nearly spent")
    assert msgs[-1]["content"][0]["type"] == "tool_result" and _alternates(msgs)
    assert "Budget nearly spent" not in str(fake.calls[54]["messages"])


def test_forced_submit_on_token_cap(bundle_fixture):
    cfg = _final_cfg(max_input_tokens_total=1000, compact_after_input_tokens=500)
    fake = FakeAnthropic([response([_stats(1)], "tool_use", 600, 1), _sub()])
    AnthropicEditor(cfg, client=fake, rollout_fn=NOROLL).propose(bundle_fixture, [])
    assert [t["name"] for t in fake.calls[1]["tools"]] == ["submit_edit_plan"]
    assert "Budget nearly spent" in str(fake.calls[1]["messages"][-1])


def test_forced_invalid_plan_retries_once_then_gives_up(bundle_fixture):
    cfg = _final_cfg(max_tool_calls_per_iteration=5)  # reserve 5: final from the first call
    fake = FakeAnthropic([_sub("a", BAD), _sub("b", GOOD)])
    AnthropicEditor(cfg, client=fake, rollout_fn=NOROLL).propose(bundle_fixture, [])
    block = fake.calls[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True and [t["name"] for t in fake.calls[1]["tools"]] == ["submit_edit_plan"]
    fake = FakeAnthropic([_sub("a", BAD), _sub("b", BAD)])
    with pytest.raises(EditorGaveUp, match="final mode"):
        AnthropicEditor(cfg, client=fake, rollout_fn=NOROLL).propose(bundle_fixture, [])
    assert len(fake.calls) == 2
