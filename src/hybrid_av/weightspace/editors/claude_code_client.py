"""`client.messages.create` stand-in that calls the Claude Code CLI (`claude -p`) on the user's subscription.

AnthropicEditor accepts an injected client; this one renders tool schemas and the transcript into a prompt and
parses one JSON tool call back. `--bare` is not used (it forces API-key auth). No key is read, written or logged.
"""

import json
import os
import subprocess
import tempfile
import uuid
from types import SimpleNamespace

PROTOCOL = """

## Reply protocol
You cannot call tools natively here. Reply with exactly one tool call as a single JSON object and nothing else:
{"name": "<tool name>", "input": {<arguments matching the tool's input_schema>}}
Use only the tools listed under "Available tools"."""


class ToolUse:
    type = "tool_use"

    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id, self.name, self.input = id, name, input

    def model_dump(self) -> dict:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


class Response:
    def __init__(self, stop_reason: str, content: list, input_tokens: int, output_tokens: int) -> None:
        self.stop_reason, self.content = stop_reason, content
        self.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)

    def model_dump(self) -> dict:
        return {
            "stop_reason": self.stop_reason,
            "content": [b.model_dump() for b in self.content],
            "usage": vars(self.usage).copy(),
        }


def _render_block(b) -> str:
    b = b.model_dump() if hasattr(b, "model_dump") else b
    if isinstance(b, str):
        return b
    if b["type"] == "text":
        return b["text"]
    if b["type"] == "tool_use":
        return "Tool call: " + json.dumps({"name": b["name"], "input": b["input"]})
    if b["type"] == "tool_result":
        tag = "Tool error" if b.get("is_error") else "Tool result"
        return f"{tag} (call {b['tool_use_id']}): {b['content']}"
    return json.dumps(b)


def _render(tools: list[dict], messages: list[dict]) -> str:
    out = ["## Available tools"]
    out += [json.dumps({k: t[k] for k in ("name", "description", "input_schema")}) for t in tools]
    out.append("\n## Conversation so far")
    for m in messages:
        c = m["content"]
        body = c if isinstance(c, str) else "\n".join(_render_block(b) for b in c)
        out.append(f"[{m['role']}]\n{body}")
    out.append("\nReply now with exactly one tool call as JSON.")
    return "\n".join(out)


def _parse_call(text: str, names: set[str]) -> dict | None:
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("name") in names and isinstance(obj.get("input"), dict):
            return obj
    return None


class _Messages:
    def __init__(self, timeout: int, retries: int) -> None:
        self.timeout, self.retries = timeout, retries

    def create(
        self, model, max_tokens, system, tools, messages, tool_choice=None
    ) -> Response:  # no CLI flag for max_tokens/tool_choice; one offered tool is the choice
        prompt = _render(tools, messages)
        names = {t["name"] for t in tools}
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        cmd = [
            "claude", "-p", "--model", model, "--output-format", "json",
            "--system-prompt", system + PROTOCOL, "--tools", "", "--no-session-persistence",
            "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
        ]  # fmt: skip
        tin = tout = 0
        for _ in range(self.retries + 1):
            # cwd is an empty temp dir so no project CLAUDE.md is discovered
            with tempfile.TemporaryDirectory() as cwd:
                try:
                    p = subprocess.run(
                        cmd,
                        check=False,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        env=env,
                        cwd=cwd,
                        timeout=self.timeout,
                    )
                except subprocess.TimeoutExpired as e:
                    raise RuntimeError(f"claude CLI timed out after {self.timeout}s") from e
            last = (p.stderr.strip().splitlines() or [""])[-1]
            if p.returncode != 0:
                raise RuntimeError(f"claude CLI exit {p.returncode}: {last}")
            try:
                data = json.loads(p.stdout)
            except ValueError as e:
                raise RuntimeError(f"claude CLI gave non-JSON output: {last}") from e
            if data.get("is_error"):
                raise RuntimeError(f"claude CLI error: {str(data.get('result'))[:200]}")
            u = data.get("usage", {})
            tin += sum(
                u.get(k, 0)
                for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            )
            tout += u.get("output_tokens", 0)
            if call := _parse_call(data.get("result") or "", names):
                return Response(
                    "tool_use",
                    [ToolUse("toolu_" + uuid.uuid4().hex[:16], call["name"], call["input"])],
                    tin,
                    tout,
                )
            if names == {"record_notes"} and data.get(
                "result"
            ):  # compaction: accept plain markdown, no retry
                return Response(
                    "tool_use", [ToolUse("toolu_notes", "record_notes", {"notes": data["result"]})], tin, tout
                )
        return Response("end_turn", [], tin, tout)


class ClaudeCodeClient:
    def __init__(self, timeout: int = 600, retries: int = 2) -> None:
        self.messages = _Messages(timeout, retries)
