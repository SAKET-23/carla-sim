| method | family | n | driving score | completion | failure rate | complete frac | collisions |
|---|---|---|---|---|---|---|---|
| base_policy | merge_deadlock | 50 | 49.8 [44.1, 56.1] | 0.505 [0.446, 0.564] | 0.840 [0.740, 0.940] | 0.160 [0.060, 0.260] | 0 |
| base_policy | nominal | 50 | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0 |
| base_policy | wet_curve | 50 | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| expert | merge_deadlock | 50 | 48.7 [43.5, 54.8] | 0.489 [0.442, 0.549] | 0.860 [0.760, 0.940] | 0.140 [0.060, 0.240] | 0 |
| expert | nominal | 50 | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0 |
| expert | wet_curve | 50 | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |
| mpc_only | merge_deadlock | 50 | 64.6 [54.3, 74.9] | 0.734 [0.657, 0.812] | 0.480 [0.340, 0.620] | 0.520 [0.380, 0.660] | 24 |
| mpc_only | nominal | 50 | 86.8 [78.8, 94.0] | 0.908 [0.846, 0.961] | 0.240 [0.120, 0.360] | 0.820 [0.700, 0.920] | 8 |
| mpc_only | wet_curve | 50 | 100.0 [100.0, 100.0] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 |

## Base policy provenance (Phase 3, Task 12)

- policy_hash: `9c138d3a97d271e24523c71907712290bc7b365a08ff578481017190db6df4d8` (sha256 over the sorted state_dict, `hybrid_av.policy.io.policy_hash`, of `actor_final.pt`)
- Recipe: behaviour cloning of `ExpertPolicy` followed by 8 DAgger rounds; no PPO fine-tuning (decision log, 2026-11-18, Task 12b)
- Wall-clock: BC 986.6 s, DAgger 2529.7 s, PPO 0 s
- Total environment steps: 1,100,000 (BC 300,000 + DAgger 8 x 100,000); PPO 0
- `train_log.csv` holds only its header (no PPO iterations). `training_curve.png` therefore plots the DAgger per-round validation loss from `train_metrics.json` (final 0.0230), not return/completion against steps; it was drawn with matplotlib directly because `eval/plots.py` only has `plot_episode`.

Reproduce:

```
uv run python scripts/train_ppo.py --config configs/ppo/litesim_dagger_base.yaml --envs 8 --bc-steps 300000 --dagger-rounds 8 --dagger-steps 100000 --total-steps 0 --launch-weight 1
uv run python scripts/eval_policy.py --bench configs/eval/phase3_litesim.yaml --policy <run>/actor_final.pt
```

### Exit numbers (development seeds 50000+, 50 per family)

| Criterion | Required | Measured |
|---|---|---|
| nominal route completion / collisions | 1.0 on every seed, 0 collisions | 50/50, 0 collisions |
| nominal mean driving score | >= 90 | 100.0 |
| wet_curve failure rate | >= 0.80 | 1.00 |
| merge_deadlock failure rate | >= 0.80 | 0.84 |

### Selection across training seeds

Rule (decision log, "Base-policy selection across training seeds"): train the DAgger recipe at seeds 0-2 and launch weights 1 and 10; pick one candidate using training-range seeds only (nominal 9900-9999, wet_curve and merge_deadlock 9950-9999); run the development exit bench once on the pick. The pick is w1_s1: fewest nominal collisions (tied with w1_s0 at 0), 2 incomplete, and the best nominal driving score (98.44) with wet 1.00.

| candidate (weight_seed) | nominal collisions | nominal incomplete | nominal driving score | wet_curve failure | merge_deadlock failure |
|---|---|---|---|---|---|
| w1_s0 | 0 | 2 | 98.25 | 1.00 | 0.90 |
| **w1_s1 (pick)** | 0 | 2 | 98.44 | 1.00 | 0.86 |
| w1_s2 | 2 | 5 | 96.32 | 1.00 | 0.50 |
| w10_s0 | 1 | 3 | 97.87 | 1.00 | 0.64 |
| w10_s1 | 3 | 6 | 95.38 | 1.00 | 0.58 |
| w10_s2 | 4 | 8 | 93.67 | 1.00 | 0.48 |

Launch weight 10 was worse on every seed, so the default is now 1.
