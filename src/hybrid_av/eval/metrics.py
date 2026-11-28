"""Metric definitions from Plan/07-evaluation/metrics.md (the base metrics are unmasked; the *_ss variants use steady_state_mask)."""

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


def steady_state_mask(trace: dict[str, np.ndarray], scenario, dt: float = 0.05) -> np.ndarray:
    """True = step kept. Drops the first 5 s, junction windows (+-5 m), brake_flag >= 0.5 and MPC fallbacks."""
    n = len(trace["route_s"])
    t = trace["t"] if "t" in trace else np.arange(n) * dt
    keep = (t >= 5.0) & (trace["brake_flag"] < 0.5) & (trace["mpc_status"] < 2)  # status >= 2: fallback_*
    for entry, exit_ in scenario.junction_s:
        keep &= ~((trace["route_s"] >= entry - 5.0) & (trace["route_s"] <= exit_ + 5.0))
    return keep


def steady_state_metrics(trace: dict[str, np.ndarray], scenario, dt: float = 0.05) -> dict[str, float]:
    """Tracking metrics on the steady-state steps; NaN when too few steps are kept."""
    m = steady_state_mask(trace, scenario, dt)
    nan = float("nan")
    k = int(m.sum())
    return {
        "cross_track_rmse_ss": cross_track_rmse(trace["cross_track"][m]) if k else nan,
        "heading_rmse_deg_ss": heading_rmse_deg(trace["heading_err"][m]) if k else nan,
        "steering_jerk_rms_ss": _segment_jerk_rms(trace["delta"], m, dt),
        "mask_fraction": float(1.0 - m.mean()) if len(m) else nan,
    }


def _segment_jerk_rms(delta: np.ndarray, keep: np.ndarray, dt: float) -> float:
    """Jerk RMS pooled over contiguous kept runs only; differencing across a masked gap would add a fake spike."""
    edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(np.int8), 0]))
    j = [np.diff(delta[a:b], n=3) / dt**3 for a, b in zip(edges[::2], edges[1::2]) if b - a >= 4]
    j = np.concatenate(j) if j else np.empty(0)
    return float(np.sqrt(np.mean(j**2))) if len(j) else float("nan")
