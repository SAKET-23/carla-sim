### Iteration 1, compaction 1 (529800 input tokens)
# Notes: wet_curve weight-edit task

## Setup / constraints
- 24-64-64-3 MLP. Max 208 edits, value within 3.0 x max(1, max|w| of tensor). run_rollout allowed seeds 50000-50009 only. Budget 40 rollout episodes (6 used so far: 2 runs x 3 seeds). 60 tool calls max.
- Feature indices: 5=target_curvature, 6=friction_est, 7=surface_roughness, 9=grade? (feature idx 9 = lead_dist per spec order; attribution showed h2 units strongly tied to feature 9, 14, 11, 20, 23).

## Findings
- activation_stats: nominal h1 dead = [0,10,27,45,54,60]; h2 dead nominal = [0,7,17,23,26,29,35,36,46]. Failure h1 dead has many more units (25); h2 dead 24 units. h1 unit 0 is dead on nominal but active (frac 1.0) in failure_window. h2 unit 0 dead on both nominal and failure.
- output_attribution failure_window: alpha_v (out 0) top contributors h2: 24 (+1.92), 52 (-1.66), 44 (+1.01), 61 (+0.96), 32 (-0.89), 21 (+0.68); bias 0.4588. delta_y (out 1) contributions ~0 (basically unused). brake (out 2) top: 52 (+3.89), 43 (-1.78), 62 (+1.63), 24 (+1.54), 25 (-1.48); bias -2.74; brake saturated at about -0.999 (tanh), so brake not responsive.
- neuron_feature_attribution h2 failure_window: (19, f9) -3.79, (27, f9) -1.83, (43, f14) +1.57; h1: (30,f9) 1.37, (33,f20) 1.24, (53,f20) 1.22, (43,f23) 1.19.
- fc3.weight row 0 key values: unit 24 = 0.5298, 52 = -0.2614? (see tool output), unit 4 = 1.3088, unit 17 = -1.5568, unit 61 = 1.6898, unit 44 = 0.9237.
- counterfactual seed 0 step 50, friction_est sweep 0.2..1.0: alpha_v ~0.67-0.73 (barely sensitive to friction_est). Seed 50000 step 150: surface_roughness sweep 0..1 gives alpha_v 0.19-0.33 (non-monotone, weak); target_curvature sweep -0.1..0.2 gives alpha_v 0.40->0.35 (weak). So the policy has little sensitivity to friction/curvature in the failure state: it needs a new speed-reducing pathway.

## Plans tried
1. fc3.weight[0,24] scale 0.3: offline fail_mean_shift alpha_v -0.552 BUT nominal mean abs shift 0.57 (max 0.79) -> would break nominal retention. Not rolled out. Rejected.
2. fc1.weight[0,7]=3.0 + fc2.weight[24,0]=-1.5: fail alpha_v shift -0.229, nominal shift 0.0. Plan with fc2[24,0]=-3.0: fail shift -0.247, nominal 0. Rollout seeds 50000-2: 50000 FAIL (lat ratio 1.297), 50001 pass (0.472), 50002 FAIL (1.138). Max lat ratios identical to baseline -> effect too small to matter.
3. fc1[0,7]=10, fc2[24,0],[44,0],[61,0],[21,0]=-3: fail shift -0.291, nominal 0. Rollout: same failures with identical lat ratios (1.297, 0.472, 1.138), AND route completion dropped to ~0.53-0.59 with 1370 steps (timeouts/too slow) -> bad; overly slow elsewhere in episode while not fixing the curve peaks. Rejected.
4. fc1[0,7]=10, fc2[0,0]=3, fc3[0,0]=-5 (new path via dead h2 unit 0): fail shift -0.292, delta_y -0.017, nominal 0. Not rolled out.
5. Same plus fc1[0,6]=-6: shift unchanged -0.292. Not rolled out.
- No static gate failures seen (all gates passed); one tool error: rollout seeds must be 50000-50009; counterfactual step must be < episode length (241 for seed 50000 trace).

## Observations / hypothesis
- h1 unit 0 fires on failure states but alpha_v shift saturates ~-0.29 regardless of routing strength, so unit 0's activation is probably small/limited or the shift is capped by something (h2 unit 0 may be clipped by relu inputs; h1 unit 0 activation magnitude on failure states is probably small). Detector needs a better trigger: a feature that actually separates the failure window (curve + wet) from nominal, e.g. weighting target_curvature (f5) and friction_est (f6) with proper sign, plus a bias to set threshold; fc1.bias[0] is -0.0385.
- Rollout lat-ratio values are unchanged vs baseline, meaning the speed reduction did not occur at the critical curve steps; need to check h1 unit 0 activation at the peak-lat-accel steps and the failure/nominal separation of candidate features (compare z for failure vs nominal at the max lat accel step).

## Next steps
- Inspect failure vs nominal feature values (z) at curve steps (friction_est, target_curvature, vx) to pick a discriminating trigger; use get_tensor/activation stats to read h1 unit 0 activation in the failure window.
- Build detector from unit 0 (or another nominal-dead h1 unit like 10, 27, 45, 54, 60) with weights on discriminating features and suitable bias, route to dead h2 unit(s) (0, 7, 17, 23, 26, 29, 35, 36, 46 are dead on nominal) with a negative fc3 weight on alpha_v (target shift about -0.5 or more on failure states only, nominal shift 0).
- Verify offline (nominal shift ~0, fail shift large) then rollout on seeds 50000-50002 and a few more before a single submit_edit_plan.
- Avoid slowing the whole episode (route completion/timeouts) and avoid changing fc3[0,24] globally (nominal break).

### Iteration 1, compaction 2 (525762 input tokens)
# Notes: wet_curve weight-edit task (iteration 1, compaction 2)

## Setup / constraints
- 24-64-64-3 MLP (relu, relu, tanh). Max 208 edits, value within 3.0 x max(1, max|w| of tensor). run_rollout allowed seeds 50000-50009 only. Budget 40 rollout episodes (9 used so far: 3 runs x 3 seeds). 60 tool calls max (many used; be economical).
- Feature indices per spec order: 0=vx, 1=vy, 2=yaw_rate, 3=lateral_offset, 4=heading_error, 5=target_curvature, 6=friction_est, 7=surface_roughness, 8=grade, 9=lead_dist, ... 14=ped_dist, 20=intersection_gap, 23=stop_dist.

## Findings
- activation_stats nominal: h1 dead = [0,10,27,45,54,60]; h2 dead = [0,7,17,23,26,29,35,36,46]. Failure_window: h1 dead has 30 units, h2 dead 31 units. h1 unit 0 is dead on nominal but active (frac 1.0) in failure window; h2 unit 0 dead on both nominal and failure. h2 units 52, 62, 24, 59, 43 have huge mean activations (3-6).
- output attribution failure_window: alpha_v top contributors h2: 24 (+1.92), 52 (-1.66), 44 (+1.01), 61 (+0.96), 32 (-0.89). delta_y basically unused. Brake saturated near -0.999 (not responsive).
- fc1.weight row 0 values (features 0..23): [0.0851,-0.0765,-0.0038,0.094,0.1515,0.171,-0.1194,0.2061,-0.0488,-0.0528,0.1626,0.0065,-0.1692,-0.1197,-0.0159,-0.0061,0.0187,-0.0444,-0.0101,0.0474,0.0579,0.1516,0.1893,0.099]; fc1.bias[0] = -0.0385. fc1.weight column 0 (vx) is mostly O(1) across units (e.g. unit 44 = 2.75). fc2.weight row 0 small (~0.1). fc2.bias[0] = -0.1309. fc3.weight[0,:]: unit 0 = -0.0311, 4=1.3088, 17=-1.5568, 24=0.5298, 44=0.9237, 61=1.6898.
- counterfactuals: policy has weak sensitivity to friction_est, surface_roughness, target_curvature in the failure state (alpha_v ~0.35-0.73 range). At seed 50000 step 100, feature 0 (vx) sweep: alpha_v 0.52 at vx 0.5, 0.63 at 1.0, 0.45 at 1.5, falls to -0.78 at 2.0, -1.0 at >=3 (brake flips to +1.0 for vx>=0.5 in those inputs). So vx input in raw units has a strong effect at high values: policy is vx-sensitive near 1.5-2.0 (z-space).

## Plans tried (offline unless noted)
1. fc3[0,24] scale 0.3: fail alpha_v shift -0.552, nominal mean abs 0.57 -> rejected.
2. fc1[0,7]=3 + fc2[24,0]=-1.5/-3: fail shift -0.23/-0.25, nominal 0. Rollout 50000-2: 50000 FAIL (lat 1.297), 50001 pass (0.472), 50002 FAIL (1.138) -> no change vs baseline.
3. fc1[0,7]=10, fc2[24,0],[44,0],[61,0],[21,0]=-3: shift -0.29, nominal 0; rollout identical lat ratios and route completion dropped to ~0.55 (1370 steps) -> rejected.
4. fc1[0,7]=10, fc1.bias[0]=0.3, fc2[0,0]=3, fc3[0,0]=-5: fail -0.292, nominal 0. ROLLOUT 50000-2: 50000 FAIL (1.294), 50001 pass (0.473), 50002 FAIL (1.138), route completion 0.53-0.59, steps 1370 -> no improvement and slow; rejected. Adding fc1[0,5]=10, fc1[0,6]=-6: same -0.2926.
5. Route via h2 unit 17 (fc2[17,0]=3, fc2.bias[17]+1): fail -0.292, nominal ~0. With bias +4 (value 3.856 > limit 3: gate failed): fail -0.36, nominal max 1.0. With bias set 2.9: fail -0.324, nominal mean abs 0.034, max 0.93 (leaks).
6. vx-driven unit 0 (fc1[0,0]=3, fc2[0,0]=3, fc3[0,0]=-5): bias -3/-1.8/-0.3 -> fail shift ~+0.003 (silent on both); bias 0.4 -> fail -0.007, nominal ~0; bias 0.7 -> fail -0.195, nominal mean abs 0.107 (max 0.94, leaks); bias 1.0 -> fail -0.61, nominal 0.30 (leaks). vx alone does not separate failure from nominal.
7. vx(3) + friction(-4) + bias 0.4: fail -0.05, nominal 0.
8. friction-only: fc1[0,6]=-3, bias 0: fail -0.087, nominal 0; fc1[0,6]=-10: fail -0.098, nominal 0.
9. fc1[0,7]=10, fc1[0,6]=-10, bias 0, fc2[0,0]=3, fc2.bias[0]=1.0, fc3[0,0]=-5: fail -0.893 but nominal mean abs 0.83 (constant offset from h2 bias lift leaks everywhere) -> rejected.
- Static gates only failed once (value_range for fc2.bias[17]=3.856 > 3).

## Observations / hypothesis
- Unit 0 detector on friction/roughness fires on failure states but delivered shift is capped ~-0.09 to -0.29; the h1 unit 0 activation on failure states seems small and h2 unit 0 pre-activation is negative (bias -0.13), so little gets through. Lifting h2 bias gives a large shift but also leaks to nominal (it is a constant offset since h2 unit 0 then fires everywhere).
- Key problem: offline fail-mean shift is averaged over failure-window states, and rollout lat ratios stayed identical to baseline, so slowdown does not happen at the actual critical curve steps (or is too weak there). Slowing across the whole episode hurts route completion (timeouts).
- No discriminating feature found yet; need data on z values of failure vs nominal at peak lat-accel steps (friction_est, target_curvature, vx, roughness).

## Next steps
- Inspect failure vs nominal z feature distributions at peak lat-accel steps (e.g. via get_tensor/trace or counterfactual) to find a clean separating feature (probably low friction_est combined with high |target_curvature| and high vx).
- Build detector with a large weight on h1 unit 0 (dead on nominal) combining those features with a bias set so nominal pre-activation is negative; give h2 unit 0 a strong gain (fc2[0,0] up to 3 and optionally multiple h1 dead units 10, 27, 45, 54, 60 feeding h2 unit 0 or other dead h2 units) so the signal after the h2 bias (-0.13) is large, with fc3 weight -5 on alpha_v; avoid constant bias lifts.
- Verify offline (nominal shift ~0, fail shift large), then rollout seeds 50000-50002 (and a few more), then submit_edit_plan once.

