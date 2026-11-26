"""Metric definitions from Plan/07-evaluation/metrics.md (unmasked; steady-state masking arrives in G2)."""

import numpy as np

G = 9.81


def cross_track_rmse(cross_track: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(cross_track))))


def heading_rmse_deg(err: np.ndarray) -> float:
    return float(np.degrees(np.sqrt(np.mean(np.square(err)))))


def steering_jerk_rms(delta: np.ndarray, dt: float) -> float:
    j = np.diff(delta, n=3) / dt**3
    return float(np.sqrt(np.mean(j**2))) if len(j) else 0.0


def solve_time_percentiles(solve_ms: np.ndarray) -> dict[str, float]:
    return {"p50": float(np.percentile(solve_ms, 50)), "p99": float(np.percentile(solve_ms, 99))}


def max_lat_accel_ratio(vx: np.ndarray, r: np.ndarray, mu_true: np.ndarray) -> float:
    return float(np.max(np.abs(vx * r) / (mu_true * G)))


def ood_failed(result, family: str) -> bool:
    """Failure criteria of Plan/02-system-design/scenarios.md sections 2-3 (nominal: any infraction or incomplete)."""
    inf = result.infractions
    if family == "wet_curve":
        return (
            any(v > 0 for k, v in inf.items() if k.startswith("collision_"))
            or inf["off_road_m"] > 0
            or result.max_lat_accel_ratio > 1.0
            or result.slack_friction_steps > 10
            or result.route_completion < 1  # parking in the warning zone is not success
        )
    if family == "merge_deadlock":
        return result.deadlock or result.route_completion < 1
    return any(v != 0 for v in inf.values()) or result.route_completion < 1
