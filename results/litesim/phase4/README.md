# Phase 4 results (LiteSim exit check, scripted editor)

Base policy: `results/litesim/phase3/actor_final.pt`, policy_hash `9c138d3a97d27190...` (see `../phase3/report.md`).
Config: `configs/weightspace/default.yaml` (max_iterations 5, OOD success threshold 0.8, retention 0.95, <= 208 changed values).

## Commands

```
for s in wet_curve merge_deadlock; do
  uv run python scripts/export_bundle.py --policy results/litesim/phase3/actor_final.pt --scenario $s --out runs/bundles
  uv run python scripts/run_edit_loop.py --bundle runs/bundles/bundle_${s}_9c138d3a --editor scripted --out runs/edits/${s}_scripted
done
```

Bundles: wet_curve 10/10 failing episodes, merge_deadlock 9/10 (the Phase 3 claim that the untouched policy fails both OOD scenarios holds).

## Outcome: scripted editor accepted on wet_curve (after the 2026-11-18 wet-curve fixes), not on merge_deadlock

Fixes applied (decision log 2026-11-18): `ood_failed` for wet_curve also fails an episode that does not complete the route; the scripted wet_curve edit routes to brake (`SCENARIO_OUTPUTS["wet_curve"] = ((2, +0.5),)`) through one h1 unit gated on roughness (feature 7) and speed (feature 0) with gain 3.0, and lowers the speed threshold by 0.5 m/s per failed gate (start 7.0 m/s). Commands for wet_curve as above, plus:

```
uv run python scripts/run_edit_loop.py --bundle runs/bundles/bundle_wet_curve_9c138d3a --editor random_sparse \
    --reference-plan runs/edits/wet_curve_scripted/edit_plan.json --seed 0 --draws 10 --out runs/edits/wet_curve_random
```

wet_curve: iteration 1 (threshold 7.0 m/s) OOD success 0.400, iteration 2 (6.5 m/s) 0.875, accepted. Changed values (5 nonzero, 93 total edits): fc1.weight[0,0]=3.0, fc1.weight[0,7]=3.0, fc1.bias[0]=0.9, fc2.weight[0,0]=1.0, fc3.weight[2,0]=3.0 (rest zeroing). Nominal: 0 new collisions, 0 MPC fallbacks.
merge_deadlock: not accepted (unchanged from the earlier run: 5 iterations, OOD success 0.550; see R-32).

| Family | Editor | Accepted | Iterations | Changed params | Fraction | Retention | OOD success (need 0.8) | Input tokens | Env steps (gates + rollouts) |
|---|---|---|---|---|---|---|---|---|---|
| wet_curve | scripted | yes (iteration 2) | 2 | 93 | 0.0156 (93/5955) | 1.000 | 0.875 | 0 | 252463 + 0 |
| merge_deadlock | scripted | no (max_iterations) | 5 | 93 (rejected plan) | 0.0156 (93/5955) | 1.000 | 0.550 (18 collisions) | 0 | 166241 + 0 |
| wet_curve | random_sparse (10 draws) | 0 of 10 accepted | - | 93 each | - | - | - | 0 | - |
| merge_deadlock | random_sparse (10 draws) | not run (no accepted scripted plan to match) | - | - | - | - | - | 0 | - |
| wet_curve | Claude (sonnet, CLI) | no (token_cap during iteration 3) | 2 plans + 1 unfinished | 1 / 13 | 1/5955, 13/5955 | 0.849 / 0.878 (fails 0.95) | 0.975 / 0.925 | 3,005,656 (out 134,878) | 343601 + 52511 |
| merge_deadlock | Claude (sonnet, CLI) | no (max_iterations) | 5 | 4 / 7 / 2 / 5 / 2 | up to 7/5955 | 0.940 (iter 5; 1.0 where reported otherwise) | 0.525 / 0.475 / 0.550 / 0.550 / 0.125 | 3,010,806 (out 78,936) | 617281 + 189917 |
| wet_curve | Claude (sonnet, CLI) + compaction | no (editor_gave_up: max_tool_calls_per_iteration 60, iteration 1) | 0 plans | - | - | - | - | 1,190,681 (out 39,236) | 0 + 10068 |
| wet_curve | Claude (sonnet, CLI) + compaction + forced submit | no (max_iterations) | 5 plans | 9 / 9 / 9 / 10 / 8 | up to 10/5955 | 1.000 / 1.000 / 0.903 / 0.939 / 1.000 | 0.000 in all 5 (iteration 5: 32 lat_accel_ratio failures, others did not finish the route) | 2,532,322 (out 142,639) | 725466 + 87511 |

Random-control acceptance count: 0 of 10 (wet_curve).

`wet_curve_scripted/` holds the accepted `summary.json`, `edit_plan.json`, `policy_edited.pt` and the accepted iteration's `plan.json`, `gates.json`, `eval_summary.json`; `merge_deadlock_scripted/` holds the last rejected iteration.

## Claude editor run (Task 10 Step 4, 2026-11-19): NOT accepted on either family

Run through the Claude subscription (`claude -p`, model `sonnet`, `ClaudeCodeClient`, `configs/weightspace/claude_code.yaml`; decision log 2026-11-19), gates and thresholds unchanged:

```
uv run python scripts/run_edit_loop.py --bundle runs/bundles/bundle_<family>_9c138d3a --editor anthropic --config configs/weightspace/claude_code.yaml --out runs/edits/<family>_claude
```

Each run builds its own editor, so the 3M input-token cap (cache-creation and cache-read tokens counted, since the full transcript is re-sent on every call) applies per family run.

- wet_curve: iteration 1 (1 changed value) OOD 0.975 but retention 0.849; iteration 2 (13 values) OOD 0.925 but retention 0.878 (0 new collisions, 0 MPC fallbacks in both). Iteration 3 hit the token cap (3,005,656 input tokens) before submitting a plan, stop_reason `token_cap`. Unlike the scripted edit, Claude's edits fixed the OOD failure but cost nominal retention.
- merge_deadlock: 5 iterations, every plan failed `ood_success` (0.525, 0.475, 0.550, 0.550, 0.125) and `no_new_collisions` (3, 4, 2, 3, 4 new nominal collisions); iteration 5 also failed retention (0.940). stop_reason `max_iterations`.
- Every iteration shows analysis tool calls (activation_stats, attribution, get_tensor, evaluate_plan_offline, run_rollout) before `submit_edit_plan` (18 to 48 calls). Raw `messages.jsonl` transcripts (~0.6 MB each) are not committed; per-iteration `plan.json`, `gates.json`, `eval_summary.json`, `usage.json` and `summary.json` are in `wet_curve_claude/` and `merge_deadlock_claude/`.
- Phase 4 exit criterion (Claude editor accepted on >= 1 family) is NOT met. Per Task 10 Step 4 this is stopped for an orchestrator decision; `max_iterations` and the token cap were not raised.

## Claude editor with transcript compaction (2026-11-19), wet_curve only: NOT accepted (no plan submitted)

Same command and `configs/weightspace/claude_code.yaml` (threshold 500k input tokens, cap 3M, gates unchanged, fresh notes). Iteration 1 made 2 compactions (at 529,800 and ~500k input tokens; notes in `wet_curve_claude_compact/notes.md`), then the editor reached `max_tool_calls_per_iteration` (60; 61st call refused) without calling `submit_edit_plan`. Total 1,190,681 input / 39,236 output tokens (well under the 3M cap); no plan reached the gates, so no OOD/retention numbers. Compaction worked as intended (per-call input stayed at 5k-32k tokens instead of growing to ~100k+), but the per-iteration tool-call counter is not reset by compaction, so it became the binding limit. The loop stops on `editor_gave_up`, not `max_iterations`, so iterations 2-5 did not run. `max_tool_calls_per_iteration` was not changed. Raw `messages.jsonl` left out.

## Claude editor with compaction and forced submit (2026-11-19), wet_curve only: NOT accepted (5 plans, OOD 0.000 each)

Same command and `configs/weightspace/claude_code.yaml` (cap 3M, compaction 500k, `submit_reserve_calls` 5, gates unchanged, fresh notes). Every iteration ended in a scored plan (no `editor_gave_up`), stop_reason `max_iterations`; 2 compactions (iteration 1 at 527,933 and iteration 3 at 521,047 input tokens); no random control (no accepted plan).

| Iter | Edits | OOD | Retention | Failed gates | Tool calls | Forced | Input / output tokens |
|---|---|---|---|---|---|---|---|
| 1 | 9 | 0.000 | 1.000 | ood_success | 48 | no | 815,552 / 38,063 |
| 2 | 9 | 0.000 | 1.000 | ood_success | 18 | no | 290,883 / 19,789 |
| 3 | 9 | 0.000 | 0.903 | ood_success, retention | 44 | no | 842,524 / 47,202 |
| 4 | 10 | 0.000 | 0.939 | ood_success, retention | 9 | no | 127,621 / 9,122 |
| 5 | 8 | 0.000 | 1.000 | ood_success | 21 | yes (token trigger, remaining cap below 500k) | 455,742 / 28,463 |

Totals 2,532,322 input / 142,639 output tokens (under the 3M cap). The call-budget trigger never fired (max 48 of 60 calls); only iteration 5 was forced, by the remaining-token rule. Nominal collisions 0 and MPC fallbacks 0 in all iterations. Unlike the earlier unforced run (OOD 0.975 / 0.925), these plans never fixed the OOD failure: most gate episodes ended without completing the route (empty failure reason). Notes: `notes.md`; raw `messages.jsonl` left out.
