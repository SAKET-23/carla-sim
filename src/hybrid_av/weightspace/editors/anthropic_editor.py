"""Claude weight editor: Anthropic Messages API with tool use (llm-weight-editor.md sections 3, 4, 9).

`anthropic` is imported only in `AnthropicEditor.__init__` and only when no client is injected. The API key is
read from the environment by the SDK and never touches this module.
"""

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_av.eval.metrics import ood_failed
from hybrid_av.eval.runner import run_episode
from hybrid_av.policy.network import CompactPolicy, TorchPolicySource
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.weightspace import analysis
from hybrid_av.weightspace.apply import apply_edit_plan
from hybrid_av.weightspace.bundle import Bundle, classify_failure
from hybrid_av.weightspace.editors.base import Attempt
from hybrid_av.weightspace.schema import TENSOR_SHAPES, EditPlan, WeightSpaceConfig
from hybrid_av.weightspace.validate import default_stack_factory, static_gates

SYSTEM_PROMPT = """You edit the weights of a small driving-policy network (24-64-64-3 MLP) by hand to fix a \
specific out-of-distribution failure. You may not train it. The user message holds the task description \
(architecture, features, outputs, the failure, constraints) and a summary of your earlier attempts.

Inspect activations on failure versus nominal states, find units that are dead or inactive on nominal data, and \
check an edit's effect with evaluate_plan_offline before spending rollouts. Held-out gate seeds exist that you \
cannot see; the edited policy must also keep nominal driving. When confident, call submit_edit_plan exactly once \
with a single EditPlan. Tool errors are one-line messages: read them and correct the call."""

NOTES_TOOL = {
    "name": "record_notes",
    "description": "Record your notes as markdown; the transcript is then replaced by these notes.",
    "input_schema": {"type": "object", "properties": {"notes": {"type": "string"}}, "required": ["notes"]},
}
COMPACT_REQUEST = (
    "The transcript is being compacted to save tokens. Call record_notes with the important findings as "
    "markdown: the tools you ran and what they showed (key numbers), every plan tried with its offline-eval and "
    "rollout results, failed gates seen, your current hypothesis, and the next step."
)

FINAL_NOTE = "Budget nearly spent: submit your best edit plan now with submit_edit_plan."
_STATE_SETS = ["failures", "nominal", "failure_window"]


class EditorGaveUp(Exception):
    pass


def _plan_schema() -> dict:
    return EditPlan.model_json_schema()


def _tool(name: str, description: str, props: dict, required: list[str], plan: bool = False) -> dict:
    schema = _plan_schema()
    defs = schema.pop("$defs", {})
    if plan:  # hoist $defs to the root so the nested plan's $refs still resolve
        props = {"plan": {"$ref": "#/$defs/EditPlan"}, **props}
        required = ["plan", *required]
        defs["EditPlan"] = schema
        return {
            "name": name,
            "description": description,
            "input_schema": {"type": "object", "properties": props, "required": required, "$defs": defs},
        }
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": props, "required": required},
    }


_SET = {"state_set": {"type": "string", "enum": _STATE_SETS}}
TOOLS: list[dict] = [
    _tool(
        "get_tensor",
        "Values of a policy tensor (at most 2,000 numbers). rows/cols are optional [start, stop) ranges.",
        {
            "name": {"type": "string", "enum": list(TENSOR_SHAPES)},
            "rows": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            "cols": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        },
        ["name"],
    ),
    _tool(
        "activation_stats", "Per-unit activation statistics of h1 and h2 on a state set.", _SET, ["state_set"]
    ),
    _tool(
        "neuron_feature_attribution",
        "64x24 gradient-times-input attribution of a hidden layer's units to the input features.",
        {**_SET, "layer": {"type": "string", "enum": ["h1", "h2"]}},
        ["state_set", "layer"],
    ),
    _tool(
        "output_attribution",
        "Mean contribution of each h2 unit to one output (0 alpha_v, 1 delta_y, 2 brake).",
        {**_SET, "output": {"type": "integer", "minimum": 0, "maximum": 2}},
        ["state_set", "output"],
    ),
    _tool(
        "counterfactual",
        "Policy outputs on a stored state with one feature swept over values.",
        {
            "seed": {"type": "integer"},
            "step": {"type": "integer"},
            "feature": {"type": "integer", "minimum": 0, "maximum": 23},
            "values": {"type": "array", "items": {"type": "number"}},
        },
        ["seed", "step", "feature", "values"],
    ),
    _tool(
        "evaluate_plan_offline",
        "Output shift of a plan on failure versus nominal states, plus the static gate results.",
        {},
        [],
        plan=True,
    ),
    _tool(
        "run_rollout",
        "Simulate the edited policy. scenario is the bundle's OOD scenario (bundle seeds only) or 'nominal' "
        "(rollout nominal seeds only). Each seed counts against the per-iteration rollout budget.",
        {
            "scenario": {"type": "string"},
            "seeds": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
        },
        ["scenario", "seeds"],
        plan=True,
    ),
    {
        "name": "submit_edit_plan",
        "description": "Submit the final EditPlan; ends this iteration.",
        "input_schema": _plan_schema(),
    },
]


def tools_for(cfg: WeightSpaceConfig) -> list[dict]:
    return [t for t in TOOLS if cfg.gradient_tools or t["name"] != "neuron_feature_attribution"]


def _r(x) -> list:
    return np.round(np.asarray(x, dtype=float), 4).tolist()


def _summ(x, k: int = 16) -> dict:
    """Top-k entries by magnitude (with indices) plus quantiles; arrays are never returned whole."""
    a = np.asarray(x, dtype=float)
    top = np.argsort(-np.abs(a).ravel(), kind="stable")[:k]
    return {
        "shape": list(a.shape),
        "top": [
            {"index": [int(i) for i in np.unravel_index(t, a.shape)], "value": round(float(a.ravel()[t]), 4)}
            for t in top
        ],
        "quantiles": dict(
            zip(("min", "q25", "median", "q75", "max"), _r(np.quantile(a, [0, 0.25, 0.5, 0.75, 1])))
        ),
    }


def _default_rollout_fn(cfg: WeightSpaceConfig) -> Callable[[CompactPolicy, str, int], dict]:
    if cfg.backend != "litesim":
        raise NotImplementedError(
            f"backend {cfg.backend!r}: only litesim is implemented (CARLA arrives in G5)"
        )

    def rollout(policy: CompactPolicy, scenario: str, seed: int) -> dict:
        stack = default_stack_factory()
        r = run_episode(
            LiteSim(stack.params, stack.lm), make_scenario(scenario, seed), stack, TorchPolicySource(policy)
        )
        if scenario == "nominal":
            failed, reason = ood_failed(r, "nominal"), ""
        else:
            failed, reason, _ = classify_failure(r, scenario)
        return {
            "failed": bool(failed),
            "reason": reason,
            "driving_score": round(r.driving_score, 2),
            "route_completion": round(r.route_completion, 3),
            "max_lat_accel_ratio": round(r.max_lat_accel_ratio, 3),
            "mpc_fallbacks": r.mpc_fallbacks,
            "steps": len(r.trace["t"]),
        }

    return rollout


def _dump(x):
    """JSON-safe copy of messages / blocks (SDK blocks are pydantic models)."""
    if isinstance(x, list):
        return [_dump(v) for v in x]
    if isinstance(x, dict):
        return {k: _dump(v) for k, v in x.items()}
    return x.model_dump() if hasattr(x, "model_dump") else x


def _oneline(e: Exception) -> str:
    return " ".join(str(e).split()) or type(e).__name__


class AnthropicEditor:
    name = "anthropic"

    def __init__(
        self, cfg: WeightSpaceConfig, client: Any | None = None, rollout_fn: Callable | None = None
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=5)  # SDK backoff on rate-limit/overload (spec section 9)
        self.cfg, self.client = cfg, client
        self.rollout_fn = rollout_fn or _default_rollout_fn(cfg)
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.last_messages: list[dict] = []
        self.rollout_steps = 0
        self.compactions = 0
        self.notes_dir = None  # set by run_edit_loop to the run directory; notes.md accumulates there
        self.notes = Path(cfg.notes_path).read_text() if cfg.notes_path else ""

    # ---- prompt ----
    def _first_message(self, bundle: Bundle, history: list[Attempt]) -> str:
        c = self.cfg
        constraints = (
            f"## Run constraints\nmax_edits {c.max_edits}; max_value_factor {c.max_value_factor}; retention >= "
            f"{c.retention_threshold}; OOD success >= {c.ood_success_threshold} on held-out seeds; run_rollout "
            f"budget {c.rollout_budget} episodes this iteration; at most {c.max_tool_calls_per_iteration} tool "
            "calls."
        )
        parts = [(bundle.root / "README.md").read_text(), constraints]
        if self.notes:
            parts.append("## Notes so far\n" + self.notes)
        for a in history:
            failed = [f"{g.gate}: {g.detail}" for g in a.gates if not g.passed]
            parts.append(
                f"## Previous attempt {a.iteration}\nedits: {len(a.plan.edits)}; hypothesis: {a.plan.hypothesis}\n"
                f"failed gates: {failed or 'none'}\naggregate metrics: {a.eval_summary}"
            )
        return "\n\n".join(parts)

    # ---- tools ----
    def _seed_check(self, scenario: str, seeds: list[int]) -> None:
        c, b = self.cfg, self._bundle
        if scenario == b.scenario:
            lo, hi = c.bundle_seeds
        elif scenario == "nominal":
            lo, hi = c.rollout_nominal_seeds
        else:
            raise ValueError(f"scenario must be {b.scenario!r} or 'nominal', got {scenario!r}")
        if bad := [s for s in seeds if not lo <= s <= hi]:
            raise ValueError(f"seeds {bad} not allowed for {scenario!r}; allowed {lo}-{hi}")

    def _states(self, name: str) -> np.ndarray:
        b = self._bundle
        return {"failures": b.Z_fail, "nominal": b.Z_nom, "failure_window": b.Z_window}[name]

    def _run_tool(self, name: str, a: dict) -> Any:
        b, cfg = self._bundle, self.cfg
        if name == "get_tensor":
            t = b.policy.state_dict()[a["name"]].detach().numpy()
            if a.get("rows"):
                t = t[slice(*a["rows"])]
            if a.get("cols"):
                t = t[:, slice(*a["cols"])]
            if t.size > 2000:
                raise ValueError(f"{t.size} numbers requested, at most 2000 per call; use rows/cols")
            return {"shape": list(t.shape), "values": _r(t)}
        if name == "activation_stats":
            st = analysis.activation_stats(b.policy, self._states(a["state_set"]))
            return {
                layer: {
                    "dead": v["dead"],
                    **{k: _summ(v[k]) for k in ("active_frac", "mean", "max")},
                }
                for layer, v in st.items()
            }
        if name == "neuron_feature_attribution":
            if not cfg.gradient_tools:
                raise ValueError("tool disabled")
            return _summ(
                analysis.neuron_feature_attribution(b.policy, self._states(a["state_set"]), a["layer"])
            )
        if name == "output_attribution":
            r = analysis.output_attribution(b.policy, self._states(a["state_set"]), a["output"])
            return {"contribution": _summ(r["contribution"]), "bias": round(r["bias"], 4)}
        if name == "counterfactual":
            z = b.traces[a["seed"]]["z"][a["step"]]
            return {
                "outputs": _r(analysis.counterfactual(b.policy, z, a["feature"], np.asarray(a["values"])))
            }
        if name == "evaluate_plan_offline":
            plan = EditPlan.model_validate(a["plan"])
            r = analysis.evaluate_plan_offline(b.policy, plan, b.Z_fail, b.Z_nom)
            gates = static_gates(plan, b.policy, cfg, np.concatenate([b.Z_fail, b.Z_nom]))
            r["static_gates"] = [{"gate": g.gate, "passed": g.passed, "detail": g.detail} for g in gates]
            return r
        if name == "run_rollout":
            plan = EditPlan.model_validate(a["plan"])
            self._seed_check(a["scenario"], a["seeds"])
            if len(a["seeds"]) > cfg.rollout_budget - self._rollouts_used:
                raise ValueError("rollout budget exhausted")
            self._rollouts_used += len(a["seeds"])
            edited, _ = apply_edit_plan(b.policy, plan)
            out = []
            for s in a["seeds"]:
                r = self.rollout_fn(edited, a["scenario"], s)
                self.rollout_steps += r.get("steps", 0)
                out.append({"seed": s, **r})
            return out
        raise ValueError(f"unknown or unavailable tool {name!r}")

    def _compact(self, messages: list[dict], tools: list[dict]) -> str:
        """One extra call: the model writes its findings via record_notes (or plain text); returns the notes."""
        last = messages[-1]
        ask = {"type": "text", "text": COMPACT_REQUEST}
        content = (
            [{"type": "text", "text": last["content"]}]
            if isinstance(last["content"], str)
            else last["content"]
        )
        msgs = [*messages[:-1], {**last, "content": [*content, ask]}]
        self.last_messages.append({"type": "request", "messages": _dump(copy.deepcopy(msgs))})
        resp = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens_per_call,
            system=SYSTEM_PROMPT,
            tools=[NOTES_TOOL],
            messages=msgs,
        )
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        self.last_messages.append({"type": "response", **_dump(resp.model_dump())})
        for blk in resp.content:
            if blk.type == "tool_use" and blk.name == "record_notes":
                return str(blk.input.get("notes", ""))
        return "\n".join(blk.text for blk in resp.content if blk.type == "text")

    # ---- loop ----
    def propose(self, bundle: Bundle, history: list[Attempt]) -> EditPlan:
        cfg = self.cfg
        self._bundle, self._rollouts_used, n_calls = bundle, 0, 0
        messages: list[dict] = [{"role": "user", "content": self._first_message(bundle, history)}]
        self.last_messages = []
        since = 0  # input tokens since the last compaction (the iteration start counts as one)
        tools = tools_for(cfg)
        final, final_errors, extra = False, 0, {}
        while True:
            if self.usage["input_tokens"] >= cfg.max_input_tokens_total:
                raise EditorGaveUp("token cap")
            if not final and (
                cfg.max_tool_calls_per_iteration - n_calls <= cfg.submit_reserve_calls
                or cfg.max_input_tokens_total - self.usage["input_tokens"]
                < (cfg.compact_after_input_tokens or cfg.max_input_tokens_total // 10)
            ):
                final = True  # forced submit: one note, only the submit tool offered
                tools = [t for t in TOOLS if t["name"] == "submit_edit_plan"]
                extra = {"tool_choice": {"type": "tool", "name": "submit_edit_plan"}}
                note = {"type": "text", "text": FINAL_NOTE}
                last = messages[-1]  # merge into the pending tool-result turn so roles alternate
                if last["role"] == "user":
                    c = last["content"]
                    last["content"] = [*([{"type": "text", "text": c}] if isinstance(c, str) else c), note]
                else:
                    messages.append({"role": "user", "content": [note]})
            if not final and cfg.compact_after_input_tokens and since > cfg.compact_after_input_tokens:
                at, self.compactions = since, self.compactions + 1
                notes = self._compact(messages, tools).strip()
                since = 0
                if notes:
                    entry = f"### Iteration {len(history) + 1}, compaction {self.compactions} ({at} input tokens)\n{notes}\n"
                    self.notes += ("\n" if self.notes else "") + entry
                    if self.notes_dir is not None:
                        with (Path(self.notes_dir) / "notes.md").open("a") as f:
                            f.write(entry + "\n")
                    messages = [{"role": "user", "content": self._first_message(bundle, history)}]
                continue  # re-check the token cap before the next call
            self.last_messages.append({"type": "request", "messages": _dump(copy.deepcopy(messages))})
            resp = self.client.messages.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens_per_call,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
                **extra,
            )
            self.usage["input_tokens"] += resp.usage.input_tokens
            self.usage["output_tokens"] += resp.usage.output_tokens
            since += resp.usage.input_tokens
            self.last_messages.append({"type": "response", **_dump(resp.model_dump())})
            if resp.stop_reason != "tool_use":
                raise EditorGaveUp(f"stop_reason {resp.stop_reason!r} without a submitted plan")
            uses = [blk for blk in resp.content if blk.type == "tool_use"]
            if n_calls + len(uses) > cfg.max_tool_calls_per_iteration and not final:
                raise EditorGaveUp("max_tool_calls_per_iteration")
            n_calls += len(uses)
            # blocks are replayed unchanged (thinking blocks must survive the round trip)
            messages.append({"role": "assistant", "content": list(resp.content)})
            results = []
            for u in uses:
                try:
                    if final and u.name != "submit_edit_plan":
                        raise ValueError("only submit_edit_plan is available now")
                    if u.name == "submit_edit_plan":
                        return EditPlan.model_validate(u.input)
                    content = json.dumps(self._run_tool(u.name, u.input))
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": content})
                except (ValueError, KeyError, TypeError, IndexError) as e:
                    final_errors += final
                    if final_errors > 1:  # one retry in final mode, then give up
                        raise EditorGaveUp(f"invalid plan in final mode: {_oneline(e)}") from e
                    results.append(
                        {"type": "tool_result", "tool_use_id": u.id, "content": _oneline(e), "is_error": True}
                    )
            messages.append({"role": "user", "content": results})
