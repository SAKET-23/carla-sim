# CARLA results (RunPod)

These are results from the phases that run on a rented GPU, all with CARLA 0.9.16 in synchronous mode (dt 0.05 s). Pods run on RunPod Secure, an RTX 4000 Ada with 20 GB VRAM and 16–18 vCPU at $0.28/h. The raw outputs are in each phase folder.

| Phase | Status | Folder |
|---|---|---|
| G1 pod bring-up | Done (GO) | [`g1/`](g1/) |
| G2 integration check and calibration | In progress | [`g2/`](g2/) |

## G1: pod bring-up (2026-11-27, pod time 2.1 h)

Verdict: **GO**. Full summary: [`g1/g1_summary.md`](g1/g1_summary.md).

- **Vulkan works:** headless CARLA renders with Vulkan on a RunPod pod once `NVIDIA_DRIVER_CAPABILITIES=all` is set. Smoke tests 01–04 pass, and [`g1/smoke04.png`](g1/smoke04.png) is a real rendered camera frame.
- **Full `Town03` crashes:** CARLA 0.9.16 segfaults on `load_world` every time on this pod. `Town03_Opt` has the same road network and loads fine, so it is now the default everywhere.
- **Python path fix needed:** the ROS 2 nodes run on the system Python and need the repo source plus the venv packages on `PYTHONPATH`. The pod setup now adds them.
- **Full sensor rig through ROS is slow:** about 1.6–1.8 ticks/s at both Low and Epic quality (target 20). G2 Task 0b looked for the cause.
- **Pinned dependencies:** the versions that were checked are in [`g1/pins.txt`](g1/pins.txt).

## G2: integration check and calibration

### Task 0: steady-state tracking metrics (Mac, done)

`eval/metrics.py::steady_state_mask` drops four kinds of steps from the tracking metrics:
- the first 5 s;
- junction windows, from 5 m before entry to 5 m after exit;
- steps with `brake_flag ≥ 0.5`;
- steps where the MPC fell back to its backup command.

`run_scenario.py` now also writes `cross_track_rmse_ss`, `heading_rmse_deg_ss`, `steering_jerk_rms_ss` and `mask_fraction`. The jerk is computed only over continuous runs of kept steps, so a masked gap doesn't create a fake spike.

### Task 0b: where does the time go? ([`g2/0b_tick_profile/`](g2/0b_tick_profile/tick_profile.csv))

Town03_Opt at Low quality, with the sensors driven through the ROS bridge:

| Sensors | Ticks/s | CARLA cores | Bridge cores | Pod cores |
|---|---|---|---|---|
| none, no rendering (the PPO setup) | **201.7** | 1.45 | 1.01 | 2.90 |
| 2 cameras (RGB + semantic, 1280×720) | **2.2** | 1.42 | 0.87 | 2.35 |
| + LiDAR | 1.77 | 1.76 | 0.90 | 2.72 |
| full rig (+ radars, IMU) | 1.78 | 1.77 | 0.90 | 2.74 |
| full, LiDAR at 32 channels / 600k points/s | 1.97 | 1.63 | 0.89 | 2.61 |

The cameras cause almost all of the slowdown, and the CPU and GPU are both mostly idle. More vCPUs or a bigger GPU would not help.

### Session 1 (2026-11-28, pod time 1.4 h, three CARLA servers in parallel)

Step outcomes are in [`g2/session1/status.tsv`](g2/session1/status.tsv), and test results in [`g2/session1/ic_junit.xml`](g2/session1/ic_junit.xml).

**Camera size.** These runs went through the ROS bridge, with GPU load recorded ([`camera/tick_profile.csv`](g2/session1/camera/tick_profile.csv)):

| Cameras | Ticks/s | GPU load |
|---|---|---|
| 2 × 1280×720 | 2.1 | 11 % |
| 1 × 1280×720 (RGB only) | 3.1 | 9 % |
| 2 × 960×540 | 3.2 | 16 % |
| 2 × 640×360 | 4.8 | 20 % |

**IC-40, tick rate run inside our own code, without ROS** ([`ic40/vram.csv`](g2/session1/ic40/vram.csv)):
- mode A (no rendering, no sensors): **37.1 ticks/s, PASS** (target ≥ 20);
- mode C (full rig): **7.95 ticks/s**.

Through the ROS bridge, the same full rig runs at about 2 ticks/s. So the bridge's handling of sensor messages, not CARLA's rendering, is the main cost with cameras on. The GPU stays under 20 % busy at every camera size.

**IC-39, MPC solve time on the pod CPU, measured alone** ([`mpc/bench_circle.txt`](g2/session1/mpc/bench_circle.txt)): p50 16.0 ms, p99 20.5 ms. **FAIL** against the targets of 10 ms and 20 ms. To be tuned in Task 6.

**Bugs found and fixed (in commit order on `main`):**

| Item | What happened on real CARLA | Fix |
|---|---|---|
| Map fixture | `find_map_features.py` called `world.get_traffic_lights()`, which 0.9.16 doesn't have. With no fixture, 29 integration tests skipped and the vehicle, throttle/brake and tyre-grip calibrations could not run. | Find lights with `world.get_actors().filter("traffic.traffic_light*")`, and print each curve once (`11e1b3e`) |
| IC-17 route heading | The route's heading jumped by π (assert `3.14159 < 0.3`). The planner returned waypoints behind the start, so the path doubled back. | Drop waypoints that lie behind the previous one (`c230807`) |
| IC-37 first drive | The route planner snapped the start onto a crossing road about 8 m away. The car started 90° off the route, went to full steering lock and stopped (completion 0, cross-track 8.7 m). The log also printed the goal under the label "spawn point". | Reject any route whose first waypoint is more than 5 m from the spawn, and fix the log line (`c230807`) |
| Seeds | The integration tests used benchmark seed 100000, and G2 must use only training seeds. | Use seed 0 (`26c5a1b`) |

**Map scan ([`map/`](g2/session1/map/)).**
- Curves with a radius of 28–35 m exist only in Town03_Opt: R 30.5 m over a 57° arc, and R 32.6 m over an 80° arc.
- Junctions with three roads and no traffic lights were found in Town03_Opt and Town04, and none in Town01, Town02 or Town05.
- Town03_Opt is therefore the candidate for the wet curve (R 32.6 m) and the merge (junction 1932). This departs from the plan's town rule and is logged.

**Tooling added for session 2:**
- `scripts/pick_routes.py`: picks 300–800 m routes with no lane change (10 training, 10 Town10HD);
- route sets in `carla_routes.yaml`: entry `seed % 10`, plus `--route-set` / `--route-index`;
- `CARLA_PORT` / `CARLA_TM_PORT` environment variables, so several CARLA servers can share one pod.

### Still to do in G2

Re-run on the pod with the fixes:
- the integration suite;
- vehicle, throttle/brake and tyre-grip calibration (Tasks 2–4);
- sensor positions and frames (Task 5);
- MPC re-tune (Task 6, IC-38/39);
- route selection (Task 7);
- in-process nominal runs (Task 8);
- ROS graph and parity (Task 9);
- latency (Task 10).
