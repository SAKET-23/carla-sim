"""run_episode: Sim + AutonomyStack + ActionSource -> EpisodeResult (interfaces.md section 11)."""

from dataclasses import dataclass

import numpy as np

from hybrid_av.eval.driving_score import driving_score
from hybrid_av.eval.metrics import (
    cross_track_rmse,
    heading_rmse_deg,
    max_lat_accel_ratio,
    solve_time_percentiles,
    steering_jerk_rms,
)
from hybrid_av.features.spec import I_CROSS_VEH_TTC, I_LEAD_TTC
from hybrid_av.geometry import wrap_angle
from hybrid_av.rl.expert import ActionSource
from hybrid_av.sim.base import Scenario, Sim
from hybrid_av.stack import AutonomyStack

POLICY_PERIOD = 2  # steps (10 Hz)
_STATUS = {"solved": 0, "solved_inaccurate": 1, "fallback_shift": 2, "fallback_brake": 3}
_SLACK_EPS = 1e-3
INFRACTION_KEYS = (
    "collision_pedestrian",
    "collision_vehicle",
    "collision_static",
    "red_light",
    "stop_sign",
    "off_road_m",
    "timeout",
    "blocked",
)


@dataclass
class EpisodeResult:
    scenario: str
    seed: int
    route_completion: float
    infractions: dict[str, int]
    driving_score: float
    cross_track_rmse: float
    heading_rmse_deg: float
    steering_jerk_rms: float
    mpc_solve_ms_p50: float
    mpc_solve_ms_p99: float
    mpc_fallbacks: int
    slack_friction_steps: int
    max_lat_accel_ratio: float
    min_ttc: float
    deadlock: bool
    duration: float
    trace: dict[str, np.ndarray]


def _ttc_min(feats, vx: float) -> float:
    """Lead TTC, and cross TTC only when the ego reaches the conflict point with the vehicle (metrics.md §8)."""
    ttc, cross = feats.raw[I_LEAD_TTC], feats.raw[I_CROSS_VEH_TTC]
    if abs(feats.cross_dist_conflict / max(vx, 0.1) - cross) <= 1.5:
        ttc = min(ttc, cross)
    return ttc


_DIAG_RANGE = 40.0  # m, nearest-object diagnostic columns (G2 s16 route 8: what the MPC dodged at a junction)


def _nearest_object(obs, rng: float) -> dict:
    """Nearest object within `rng` of the ego, position and velocity in the ego frame (nan if none)."""
    e = obs.ego
    c, s = np.cos(e.psi), np.sin(e.psi)
    best = min(obs.objects, key=lambda o: np.hypot(o.x - e.x, o.y - e.y), default=None)
    if best is None or np.hypot(best.x - e.x, best.y - e.y) > rng:
        return dict.fromkeys(("obj_dx", "obj_dy", "obj_vx", "obj_vy"), np.nan)
    dx, dy = best.x - e.x, best.y - e.y
    return {
        "obj_dx": c * dx + s * dy, "obj_dy": -s * dx + c * dy,
        "obj_vx": c * best.vx + s * best.vy, "obj_vy": -s * best.vx + c * best.vy,
    }  # fmt: skip


def run_episode(
    sim: Sim, scenario: Scenario, stack: AutonomyStack, source: ActionSource, record_activations=None
) -> EpisodeResult:
    if record_activations is not None:
        raise NotImplementedError("activation recording needs the Phase 3 CompactPolicy")
    obs = sim.reset(scenario)
    route = sim.route()
    stack.reset(route)
    inf = dict.fromkeys(INFRACTION_KEYS, 0)
    rows: dict[str, list] = {}
    action = None
    k = 0
    while True:
        feats = stack.features(obs)
        is_policy = k % POLICY_PERIOD == 0
        if is_policy:
            action = source(feats.z, feats)
        out = stack.act(obs, action, feats)
        cmd, ego = out.command, obs.ego
        row = {
            "t": ego.t, "x": ego.x, "y": ego.y, "psi": ego.psi, "vx": ego.vx, "vy": ego.vy, "r": ego.r,
            "delta": out.control.delta, "delta_meas": np.nan, "ax": out.control.ax,
            "steer": cmd.steer, "throttle": cmd.throttle, "brake": cmd.brake,
            "alpha_v": action.alpha_v, "delta_y": action.delta_y, "brake_flag": action.brake,
            "mu_true": obs.friction_true, "mu_est": feats.mu_est,
            "cross_track": feats.lateral - action.delta_y,
            "heading_err": wrap_angle(ego.psi - out.reference.psi[0]),
            "route_s": feats.route_s, "mpc_status": _STATUS[out.mpc.status],
            "solve_ms": out.mpc.solve_time * 1e3,
            "slack_f": out.mpc.slack_friction_max, "slack_o": out.mpc.slack_obstacle_max,
            "ttc_min": _ttc_min(feats, ego.vx),
            "policy_step": is_policy,
            **_nearest_object(obs, _DIAG_RANGE),
        }  # fmt: skip
        obs, info = sim.step(cmd)
        collision = info.collision != "none"
        events = {
            f"collision_{info.collision}": collision,
            "red_light": info.red_light_violation,
            "stop_sign": info.stop_sign_violation,
            "off_road_m": info.off_road,  # code only; the distance is accumulated below
        }
        for key, hit in events.items():
            if hit and key != "off_road_m":
                inf[key] += 1
        if info.off_road:
            inf["off_road_m"] += ego.vx * sim.dt
        row["infraction"] = next((i + 1 for i, key in enumerate(INFRACTION_KEYS) if events.get(key)), 0)
        for key, v in row.items():
            rows.setdefault(key, []).append(v)
        rows.setdefault("z", []).append(feats.z)
        k += 1
        if info.route_completed or info.route_deviation or info.blocked or info.timeout or collision:
            break
    inf["timeout"] = int(info.timeout and not info.route_completed)
    inf["blocked"] = int(info.blocked)
    tr = {key: np.asarray(v) for key, v in rows.items()}
    tr["z"] = tr["z"].astype(np.float32)
    tr["policy_step"] = tr["policy_step"].astype(bool)
    tr["mpc_status"] = tr["mpc_status"].astype(np.int8)
    tr["infraction"] = tr["infraction"].astype(np.int8)
    completion = 1.0 if info.route_completed else min(1.0, info.route_s / route.length)
    p = solve_time_percentiles(tr["solve_ms"])
    return EpisodeResult(
        scenario=scenario.name,
        seed=scenario.seed,
        route_completion=completion,
        infractions=inf,
        driving_score=driving_score(completion, inf, route.length),
        cross_track_rmse=cross_track_rmse(tr["cross_track"]),
        heading_rmse_deg=heading_rmse_deg(tr["heading_err"]),
        steering_jerk_rms=steering_jerk_rms(tr["delta"], sim.dt),
        mpc_solve_ms_p50=p["p50"],
        mpc_solve_ms_p99=p["p99"],
        mpc_fallbacks=int(np.sum(tr["mpc_status"] >= 2)),
        slack_friction_steps=int(np.sum(tr["slack_f"] > _SLACK_EPS)),
        max_lat_accel_ratio=max_lat_accel_ratio(tr["vx"], tr["r"], tr["mu_true"]),
        min_ttc=float(np.min(tr["ttc_min"])),
        deadlock=bool(info.blocked or (completion < 1 and info.timeout and tr["vx"][-1] < 0.1)),
        duration=info.sim_time,
        trace=tr,
    )
