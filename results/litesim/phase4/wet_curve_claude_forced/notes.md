### Iteration 1, compaction 1 (527933 input tokens)
# Notes: wet_curve weight edit

## Findings
- Failures: 10 episodes, all reason lat_accel_ratio (max ratio about 1.46-1.54 on rollouts). Nominal: 5 episodes.
- activation_stats: h1 dead on nominal = [0,10,27,45,54,60]; h1 unit 0 is dead on nominal AND dead on failures (failure dead list includes 10,27,45,54 but not 0 in 'failures'; 0 fully active in failure_window). h2 dead on nominal = [0,7,17,23,26,29,35,36,46]; h2 unit 0 also dead on failures.
- output_attribution alpha_v (out 0) on failures: +h2 24 (1.57), -h2 52 (-1.51), +61 (0.85), +44 (0.73), -32 (-0.66), +21 (0.54). Bias 0.4588. delta_y has essentially zero contribution. brake (out 2) dominated by h2 52 (+3.55), 43 (-1.87), 62 (+1.55), 25 (-1.31), 24 (+1.26); bias -2.7355; brake is saturated near -0.999 on counterfactual.
- fc1 attribution: h1 unit 14 has weight -3.6233 on feature 3 (lateral_offset) and is the top attributor on failures.
- fc1.weight[:,6] is friction_est column (feature 6). Counterfactual on friction_est (seed 0 step 50, values 0.3-0.9) changed alpha_v only slightly (0.69, 0.68, 0.67, 0.71), so the policy hardly reacts to friction_est.
- fc3.weight row 0 values: h2 24 = 0.5298, h2 61 = 1.6898, h2 44 = 0.9237, h2 21 = 0.8919, h2 0 = -0.0311. fc2.bias[0] = -0.1309.

## Plans tried (offline eval: fail_mean_shift alpha_v / nominal mean abs shift alpha_v)
1. fc3.bias[0] add -0.2: -0.076 / 0.079 (shifts nominal as much as failures; bad).
2. fc3.weight[0,24] and [0,61] scale 0.3: -0.748 / 0.750 (nominal shifted equally; bad).
3. Wet gate on h1 unit 0 (fc1.weight[0,6] = -3, fc1.bias[0] = 1.5) + fc2.weight[24,0], [61,0] = -1.5: -0.193 / 0.0.
4. Gate -8, bias 4 plus units 24, 61, 44, 21 at -1.5: -0.294 / 0.0, but FAILED value_range (fc1.bias limit is 3).
5. Gate -6, bias 3, units 24, 61, 44, 21 at -1.5: -0.283 / 0.0, all static gates pass. ROLLOUT on seeds 50000-50004 (allowed seeds are 50000-50009): all 5 still failed, lat_accel_ratio 1.46-1.54. Not fixed.
6. Same as 5 plus h2 unit 0 route (fc2.weight[0,0] = 3, fc3.weight[0,0] = -4), no fc2 bias change: -0.292 / 0 (h2 unit 0 stays dead).
7. Gate -6, bias 3, units at -3: -0.293 / 0.
8. Gate -10, bias 3, h2 unit 0 route (5, -5), units at -3: -0.205 / 0.
9. Gate -6, bias 3, fc2.weight[0,0] = 3, fc2.bias[0] = 1.5, fc3.weight[0,0] = -4, units 24, 61 at -1.5: -0.903 / 0.837 (h2 unit 0 fires on nominal; bad).
10. Same with fc2.bias[0] = 0.4: -0.317 / 0.088 (some nominal shift, max 0.556).
11. Gate -4, bias 3, units 24/61/44/21 at -3, plus fc2.weight[52,0] = 3 and fc3.bias[0] add -0.1: -0.882 / 0.729 (nominal shifted; bad, gate leaks onto nominal at -4 with the extra edits, brake +0.238).
12. Gate -5, bias 3, units at -3: -0.373 / 0.0.
13. Gate -4.5, bias 3, units 24/61/44/21 at -3: -0.595 / 0.0. BEST SO FAR: strong failure shift with zero nominal shift. Not yet rolled out.

## Failed gates / errors seen
- value_range: fc1.bias[0] = 4 exceeds limit 3.
- run_rollout seeds must be in 50000-50009 for wet_curve.

## Hypothesis
The policy barely reacts to friction_est, and the h1 unit 0 gate (weight on friction_est feature, bias 3) can detect wet states without touching nominal states. Lowering alpha_v when the gate fires reduces speed. Gate threshold is sensitive: -6 weak, -5 moderate, -4.5 strong with nominal untouched, -4 leaks to nominal when combined with other edits (plan 11, though plan 11 also had extra edits). Plan 5's rollout showed that a -0.28 mean alpha_v shift is not enough; a bigger shift is needed.

## Next steps
- Roll out plan 13 (gate -4.5, bias 3, units 24/61/44/21 at -3) on seeds 50000-50004 to see whether lat_accel_ratio drops below 1.0 and no new failures appear.
- If still failing, try -4.25 or -4.3 offline, checking nominal shift stays 0, and consider adding more alpha_v-raising h2 units or a fc3.bias decrease gated only on wet states.
- If the rollout passes, check a few more seeds (50005-50009) for retention/held-out safety, then submit_edit_plan once.

### Iteration 3, compaction 2 (521047 input tokens)
# Notes: wet_curve weight edit (iteration 1, compaction 2)

## Findings so far
- Failures: 10 episodes, all reason lat_accel_ratio (max ratio ~1.46-1.54 on base rollouts). Nominal: 5 episodes.
- activation_stats: h1 dead on nominal = [0,10,27,45,54,60]; h2 dead on nominal = [0,7,17,23,26,29,35,36,46]. h1 unit 0 and h2 unit 0 are dead on nominal (h2 unit 0 also dead on failures; h1 unit 0 fully active in failure_window).
- output_attribution alpha_v (out 0) on failures: +h2 24 (1.57), -h2 52 (-1.51), +61 (0.85), +44 (0.73), -32 (-0.66), +21 (0.54). Bias 0.4588. delta_y contributes ~0. Brake dominated by h2 52 (+3.55), 43, 62, 25, 24; bias -2.7355; brake saturated near -0.999.
- fc1 attribution: h1 unit 14 has weight -3.6233 on feature 3 (lateral_offset).
- fc1.weight[0,:] originally small (feature 6 weight -0.1194, feature 5 weight 0.171).
- Counterfactuals (seed 0, steps 150/200): friction_est (feat 6) values 0.0-1.2 change alpha_v only by ~0.03-0.1 (0.67-0.79); target_curvature (feat 5) 0-0.3 changes alpha_v ~0.003; yaw_rate (feat 2) hardly matters. speed_limit (feat 21) is strongly influential: values 0.5-2 give alpha_v ~0.7, 5 gives -0.25, 8+ gives -1.0 (and brake flips to +1 at speed_limit>=20). vx (feat 0) 4-16 gives alpha_v -1.0, brake +1.0 (strongly OOD-sensitive).
- failure_window h1 dead list differs from failures; fc1.weight[:,21] column read (speed_limit weights; e.g. unit 13: 1.5887, unit 22: 1.6981, unit 1: 0.8039).
- Step index limit: seed 0 episode has 305 steps in the counterfactual source.

## Plans tried (offline: fail_mean_shift alpha_v / nominal mean abs shift alpha_v) and rollouts
Gate = h1 unit 0 via fc1.weight[0,6] (friction_est) and fc1.bias[0]=3; suppress h2 units 24/61/44/21 via fc2.weight[u,0]; optionally h2 unit 0 route (fc2.weight[0,0], fc2.bias[0], fc3.weight[0,0]).
1. fc3.bias[0] add -0.2: -0.076 / 0.079 (bad).
2. fc3.weight[0,24],[0,61] scale 0.3: -0.748 / 0.750 (bad).
3-8. Various gate -3..-10 variants: shift -0.19..-0.37, nominal 0.
5. Gate -6, bias 3, units at -1.5: rollout 5 seeds all failed (1.46-1.54).
9-11. Variants with h2 unit 0 fed on nominal or gate -4 leak into nominal (nominal shift 0.09-0.84; bad).
13. Gate -4.5, bias 3, units 24/61/44/21 at -3: -0.595 / 0.
14. Plan 13 rolled out on 50000-50004: seed 50001 PASSED (0.472), 50000/50002/50003/50004 failed (1.297, 1.138, 1.287, 1.125) - modest improvement from ~1.5.
15. Plan 13 + h2 unit 0 route (fc2.weight[0,0]=3, fc3.weight[0,0]=-6, units at -3): -0.60 / 0; rollout same ratios (1.297, 0.472, 1.138, 1.287, 1.125), 765 steps on seed 50000.
16. Units at -6, fc2.weight[0,0]=6, fc2.bias[0]=0, fc3.weight[0,0]=-6: -0.646 / 0.
17. fc3.weight[0,0]=-9: FAILED value_range (limit 6.67194); also fc3.weight[0,52] scale 0 leaks nominal (0.24 shift) - do not use.
18. Plan 16 variant with fc3.weight[2,0]=3 (brake up when wet) and fc3.weight[0,0]=-6.5: offline -0.646 alpha_v, brake +0.49 on failures, nominal 0. ROLLOUT: ratios unchanged (1.294, 0.472, 1.138, 1.287, 1.125), but episodes did not finish (1370 steps, route completion ~0.53-0.6, driving score ~38-42) - stalled/slow driving; brake hurts progress but did not reduce peak lat accel ratio.
19. Two gates (h1 units 0 and 10) driving h2 unit 0: -0.58, nominal 0.24 (leak; bad), not rolled out.
20. Curvature+friction gate (fc1.weight[0,5]=10, [0,6]=-6, bias 2) + units -6 + h2 unit 0 route: offline -0.33 / 0.029 nominal mean, max 1.0. ROLLOUT: all 5 failed with WORSE ratios (1.384, 1.206, 1.284, 1.385, 1.322) and 1370 steps.
21. Diagnostic: global fc3.bias[0] add -0.4, seeds 50000/50002: ratios 1.536 and 1.447 (WORSE than base ~1.3-1.5 on these seeds ... base 1.46-1.54 range), episodes finished in 507/477 steps. So lowering alpha_v globally does NOT lower peak lateral acceleration ratio; alpha_v is probably not the effective lever (or lowering it speeds up/changes the reference generator behavior).

## Failed gates / errors seen
- value_range: fc1.bias[0]=4 exceeds limit 3; fc3.weight[0,0]=-9 exceeds 6.67194.
- run_rollout seeds must be in 50000-50009 for wet_curve.
- counterfactual step must be < episode length (305 for seed 0).
- Earlier submissions (attempts 1, 2): ood_success 0.0, nominal score 100.

## Current hypothesis
The friction_est gate works mechanically (nominal untouched, strong offline alpha_v shift) but alpha_v shifts do not translate into lower peak lat-accel ratio in rollouts; global alpha_v reduction even made ratios slightly higher. The reference generator/MPC likely maps alpha_v to a speed target in a way that makes lower alpha_v ineffective or counterproductive (possible sign convention: higher alpha_v may mean slower/more curvature-limited? need to check output_spec.json), or the peak occurs before the gate fires. Brake increase stalls driving without reducing ratio. Speed_limit (feat 21) and vx (feat 0) have much stronger effects on outputs than friction_est.

## Next steps
- Re-read output_spec.json to verify alpha_v's effect through the reference generator (sign/direction) before further edits; consider raising alpha_v in wet states instead (test with a small rollout using fc3.bias[0] add +0.3 on seeds 50000/50002).
- Investigate failure trace timing: when in the episode the peak ratio occurs vs friction_est and gate activation (use traces in failures/*.npz) and what vx/curvature are at the peak.
- Consider using speed_limit feature (strong lever) or vx-dependent gating instead of friction_est, keeping nominal shift 0 via dead units.
- Budget used: many rollouts (about 20+ episodes of 40) and a large number of tool calls; be economical. Submit one EditPlan only when a rollout shows lat_accel_ratio < 1.0 on most seeds with nominal unaffected.

