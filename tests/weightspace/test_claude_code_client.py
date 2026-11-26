import json
import subprocess
from dataclasses import replace

import pytest

from hybrid_av.weightspace.editors import claude_code_client as cc
from hybrid_av.weightspace.editors.anthropic_editor import AnthropicEditor
from hybrid_av.weightspace.schema import WeightSpaceConfig

GOOD = {
    "edits": [{"tensor": "fc3.bias", "index": [0], "op": "add", "value": -0.3, "rationale": "r"}],
    "hypothesis": "h",
    "expected_effect": "e",
}
TOOLS = [{"name": "submit_edit_plan", "description": "d", "input_schema": {"type": "object"}}]
USAGE = {
    "input_tokens": 2,
    "cache_creation_input_tokens": 100,
    "cache_read_input_tokens": 30,
    "output_tokens": 7,
}


def cli(result, rc=0, **extra):
    out = json.dumps({"is_error": False, "result": result, "usage": USAGE, **extra})
    return subprocess.CompletedProcess([], rc, out, "boom\nlast line")


@pytest.fixture
def run(monkeypatch):
    calls, replies = [], []

    def fake(cmd, **kw):
        calls.append((cmd, kw))
        return replies.pop(0)

    monkeypatch.setattr(cc.subprocess, "run", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    return calls, replies


def create(client, messages=None):
    return client.messages.create(
        model="sonnet",
        max_tokens=100,
        system="sys",
        tools=TOOLS,
        messages=messages or [{"role": "user", "content": "task"}],
    )


def test_command_env_and_tool_use_block(run):
    calls, replies = run
    replies.append(cli('Sure: {"name": "submit_edit_plan", "input": {"a": 1}}'))
    r = create(cc.ClaudeCodeClient())
    cmd, kw = calls[0]
    assert cmd[0] == "claude" and "-p" in cmd
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--tools") + 1] == "" and "--no-session-persistence" in cmd
    assert "--bare" not in cmd and "--strict-mcp-config" in cmd
    assert "ANTHROPIC_API_KEY" not in kw["env"] and "task" in kw["input"]
    b = r.content[0]
    assert r.stop_reason == "tool_use" and (b.type, b.name, b.input) == (
        "tool_use",
        "submit_edit_plan",
        {"a": 1},
    )
    assert b.id and json.dumps(r.model_dump()) and json.dumps(b.model_dump())
    assert r.usage.input_tokens == 132 and r.usage.output_tokens == 7


def test_garbage_retries_then_end_turn(run):
    calls, replies = run
    replies.extend(cli("no json here") for _ in range(3))
    r = create(cc.ClaudeCodeClient())
    assert len(calls) == 3 and r.stop_reason == "end_turn" and r.usage.input_tokens == 3 * 132


def test_nonzero_exit_raises(run):
    _, replies = run
    replies.append(cli("", rc=1))
    with pytest.raises(RuntimeError, match="last line"):
        create(cc.ClaudeCodeClient())


def test_replayed_blocks_render(run):
    calls, replies = run
    replies.append(cli('{"name": "submit_edit_plan", "input": {}}'))
    client = cc.ClaudeCodeClient()
    first = cc.Response("tool_use", [cc.ToolUse("id1", "get_tensor", {"name": "fc1.bias"})], 1, 1)
    msgs = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": list(first.content)},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "id1", "content": "oops", "is_error": True}],
        },
    ]
    create(client, msgs)
    p = calls[0][1]["input"]
    assert "get_tensor" in p and "oops" in p and "error" in p.lower()


def test_editor_with_claude_code_client(run, bundle_fixture):
    _, replies = run
    replies.append(cli(json.dumps({"name": "submit_edit_plan", "input": GOOD})))
    cfg = replace(WeightSpaceConfig(), model="sonnet")
    plan = AnthropicEditor(cfg, client=cc.ClaudeCodeClient()).propose(bundle_fixture, [])
    assert plan.edits[0].value == -0.3
