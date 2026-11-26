"""Shaped driving reward (Plan/03-models/ppo-training.md section 4)."""

from dataclasses import dataclass

import numpy as np

from hybrid_av.features.extractor import FeatureFrame
from hybrid_av.features.spec import FEATURE_NAMES
from hybrid_av.mpc.controller import MPCResult
from hybrid_av.policy.network import unscale_action
from hybrid_av.sim.base import StepInfo
from hybrid_av.types import PolicyAction

_I = {n: i for i, n in enumerate(FEATURE_NAMES)}
_G = 9.81


@dataclass(frozen=True, slots=True)
class RewardConfig:
    dt: float = 0.05
    w_progress: float = 0.05
    w_overspeed: float = -1.0
    w_lateral: float = -0.1
    lateral_scale: float = 1.75
    w_offset: float = -0.02
    w_friction: float = -1.0
    friction_frac: float = 0.8
    w_slack: float = -2.0
    slack_eps: float = 1e-3
    w_fallback: float = -5.0
    w_action_rate: float = -0.05
    w_standstill: float = -0.05
    standstill_speed: float = 0.5
    standstill_clear: float = 10.0
    r_collision: float = -50.0
    r_collision_ped_extra: float = -50.0
    r_offroad: float = -30.0
    r_red_light: float = -20.0
    r_stop_sign: float = -10.0
    r_blocked: float = -20.0
    r_timeout: float = -10.0
    r_complete: float = 20.0


def compute_reward(
    prev: FeatureFrame,
    cur: FeatureFrame,
    info: StepInfo,
    mpc: MPCResult,
    action: PolicyAction,
    prev_action: PolicyAction,
    cfg: RewardConfig,
) -> tuple[float, dict[str, float]]:
    raw = cur.raw
    vx, vlim = raw[_I["vx"]], raw[_I["speed_limit"]]
    ya, yp = unscale_action(action), unscale_action(prev_action)
    hit = info.collision != "none"
    slack = max(mpc.slack_friction_max, mpc.slack_obstacle_max) > cfg.slack_eps
    free = min(raw[_I["stop_dist"]], raw[_I["lead_dist"]]) >= cfg.standstill_clear
    terms = {
        "progress": cfg.w_progress * np.clip((cur.route_s - prev.route_s) / (vlim * cfg.dt), -1.0, 1.2),
        "overspeed": cfg.w_overspeed * max(0.0, vx - vlim) / vlim,
        "lateral": cfg.w_lateral * ((cur.lateral - action.delta_y) / cfg.lateral_scale) ** 2,
        "offset": cfg.w_offset * ya[1] ** 2,
        "friction": cfg.w_friction
        * max(0.0, abs(vx * raw[_I["yaw_rate"]]) / (cur.mu_est * _G) - cfg.friction_frac),
        "slack": cfg.w_slack * slack,
        "fallback": cfg.w_fallback * mpc.status.startswith("fallback"),
        "action_rate": cfg.w_action_rate * float(np.sum((ya - yp) ** 2)),
        "standstill": cfg.w_standstill * (vx < cfg.standstill_speed and free),
        "collision": cfg.r_collision * hit,
        "collision_ped_extra": cfg.r_collision_ped_extra * (info.collision == "pedestrian"),
        "offroad": cfg.r_offroad * (info.off_road or info.route_deviation),
        "red_light": cfg.r_red_light * info.red_light_violation,
        "stop_sign": cfg.r_stop_sign * info.stop_sign_violation,
        "blocked": cfg.r_blocked * info.blocked,
        "timeout": cfg.r_timeout * (info.timeout and not info.route_completed),
        "complete": cfg.r_complete * (info.route_completed and not hit),
    }
    terms = {k: float(v) for k, v in terms.items()}
    return sum(terms.values()), terms
