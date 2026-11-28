"""Pure fitting for the vehicle identification and friction calibration (Plan/06-integration/calibration.md §1-§3)."""

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams

G = 9.81
_R_WEIGHT = 5.0  # yaw rate counts x5 against vy in the lateral fit (calibration.md §1)


@dataclass
class Segment:
    """One recorded manoeuvre: row k is the state before `delta[k]` (front wheel angle, rad) is applied."""

    t: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    r: np.ndarray
    delta: np.ndarray
    collided: bool = False


@dataclass
class LongSamples:
    vx: np.ndarray
    ax: np.ndarray  # measured, already low-passed
    throttle: np.ndarray
    brake: np.ndarray


# --- API values ------------------------------------------------------------------------------------------------
def wheelbase_split(front, rear, com_x: float, loc, yaw_deg: float) -> tuple[float, float, float]:
    """(L, a, b) from wheel positions and the centre of mass.

    `front` and `rear` are (x, y) world positions in cm, `loc` the actor location in m, `com_x` the centre of mass
    along the heading in m relative to the actor origin, `yaw_deg` the CARLA yaw. Distances are taken along the heading."""
    # ECC-32: WheelPhysicsControl.position is a world position in cm, center_of_mass is in m from the actor origin
    h = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])

    def along(wheels):
        return float(np.mean([(np.array(w) / 100.0 - np.array(loc)) @ h for w in wheels]))

    xf, xr = along(front), along(rear)
    return xf - xr, xf - com_x, com_x - xr


def check_units(L: float, com_x: float) -> None:
    # ECC-32: a 100x error here means the cm assumption is wrong; fix the script before fitting
    if not 2.5 <= L <= 3.2 or abs(com_x) >= 1.0:
        raise ValueError(
            f"ECC-32 unit check failed: wheelbase {L:.3f} m (want 2.5-3.2), com_x {com_x:.3f} m (want < 1)"
        )


# --- lateral ---------------------------------------------------------------------------------------------------
def _rollout(
    p: VehicleParams, Cf: float, Cr: float, Iz: float, seg: Segment
) -> tuple[np.ndarray, np.ndarray]:
    """Linear dynamic bicycle model from the segment's first measured (vy, r) with its recorded delta and vx."""
    n = len(seg.t)
    dt = float(seg.t[1] - seg.t[0]) if n > 1 else 0.05
    h = dt / 2.0  # two RK4 sub-steps per sample
    a, b, m = p.a, p.b, p.m
    vy, r = float(seg.vy[0]), float(seg.r[0])
    out_vy, out_r = np.empty(n), np.empty(n)

    def f(vy, r, vx, d):
        fyf = Cf * (d - (vy + a * r) / vx)
        fyr = -Cr * (vy - b * r) / vx
        return (fyf + fyr) / m - r * vx, (a * fyf - b * fyr) / Iz

    for k in range(n):
        out_vy[k], out_r[k] = vy, r
        vx, d = max(float(seg.vx[k]), 1.0), float(seg.delta[k])
        for _ in range(2):
            k1 = f(vy, r, vx, d)
            k2 = f(vy + h / 2 * k1[0], r + h / 2 * k1[1], vx, d)
            k3 = f(vy + h / 2 * k2[0], r + h / 2 * k2[1], vx, d)
            k4 = f(vy + h * k3[0], r + h * k3[1], vx, d)
            vy += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            r += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return out_vy, out_r


def understeer_gradient(p: VehicleParams) -> float:
    return p.m * (p.b * p.Cr - p.a * p.Cf) / (p.L * p.Cf * p.Cr)


def evaluate_lateral(p: VehicleParams, segments: list[Segment], min_vx: float = 6.0) -> dict:
    ev, er = [], []
    for s in segments:
        vy, r = _rollout(p, p.Cf, p.Cr, p.Iz, s)
        ok = s.vx >= min_vx
        ev.append((vy - s.vy)[ok])
        er.append((r - s.r)[ok])
    ev, er = np.concatenate(ev), np.concatenate(er)
    return {"vy_rmse": float(np.sqrt(np.mean(ev**2))), "r_rmse": float(np.sqrt(np.mean(er**2)))}


def fit_lateral(
    segments: list[Segment], p0: VehicleParams, min_vx: float = 6.0
) -> tuple[VehicleParams, dict]:
    """Least squares over (Cf, Cr, Iz) with bounds Cf, Cr in [5e3, 2e5] and Iz in [0.5, 2] m a b (calibration.md §1)."""
    iz_ref = p0.m * p0.a * p0.b
    scale = np.array([1e4, 1e4, 1e3])
    lo = np.array([5e3, 5e3, 0.5 * iz_ref]) / scale
    hi = np.array([2e5, 2e5, 2.0 * iz_ref]) / scale
    x0 = np.clip(np.array([p0.Cf, p0.Cr, p0.Iz]) / scale, lo, hi)

    def residuals(x):
        Cf, Cr, Iz = x * scale
        res = []
        for s in segments:
            vy, r = _rollout(p0, Cf, Cr, Iz, s)
            ok = s.vx >= min_vx
            res += [(vy - s.vy)[ok], _R_WEIGHT * (r - s.r)[ok]]
        return np.concatenate(res)

    sol = least_squares(residuals, x0, bounds=(lo, hi), x_scale=1.0)
    Cf, Cr, Iz = (float(v) for v in sol.x * scale)
    fit = replace(p0, Cf=Cf, Cr=Cr, Iz=Iz)
    return fit, {**evaluate_lateral(fit, segments, min_vx), "understeer_K": understeer_gradient(fit)}


# --- longitudinal ----------------------------------------------------------------------------------------------
def lowpass(x: np.ndarray, dt: float, fc: float) -> np.ndarray:
    alpha = dt / (dt + 1.0 / (2.0 * math.pi * fc))
    y = np.empty(len(x))
    if len(x) == 0:
        return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def long_run_arrays(
    rows: np.ndarray, dt: float, skip: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(vx, ax, throttle, brake) from rows of (vx, ax, throttle, brake), dropping the first `skip` s then low-passing ax (empty arrays when the run is shorter than `skip`).

    set_target_velocity at reset makes get_acceleration spike to about v0/dt, so the transient goes before the filter."""
    a = np.asarray(rows).reshape(-1, 4)[round(skip / dt) :]
    return a[:, 0], lowpass(a[:, 1], dt, 5.0), a[:, 2], a[:, 3]


def longitudinal_problems(p: VehicleParams, lm: LongitudinalMap, r2: float) -> list[str]:
    """Reasons the longitudinal identification is not fit to be written (empty when sane)."""
    checks = [
        (0.0 <= p.rolling <= 1.0, f"rolling {p.rolling:.4g} outside [0, 1] m/s^2"),
        (0.0 <= p.drag_coeff <= 0.01, f"drag_coeff {p.drag_coeff:.4g} outside [0, 0.01]"),
        (p.max_decel > 0.0, f"max_decel {p.max_decel:.4g} <= 0"),
        (lm.throttle_gain > 0.0, f"throttle_gain {lm.throttle_gain:.4g} <= 0"),
        (lm.brake_gain > 0.0, f"brake_gain {lm.brake_gain:.4g} <= 0"),
        (lm.coast_band <= 1.0, f"coast_band {lm.coast_band:.4g} > 1 m/s^2"),
        (r2 >= 0.9, f"throttle fit R2 {r2:.3f} < 0.9"),
    ]
    return [msg for ok, msg in checks if not ok]


def lateral_problems(val: dict, gains: list[float], K: float) -> list[str]:
    """Held-out checks that failed, each named (empty when all pass)."""
    checks = [
        (val["r_rmse"] < 0.02, f"held-out r_rmse {val['r_rmse']:.4f} >= 0.02"),
        (val["vy_rmse"] < 0.1, f"held-out vy_rmse {val['vy_rmse']:.4f} >= 0.1"),
        (max(gains) < 0.10, f"steady-state gain error max {max(gains):.3f} >= 0.10"),
        (K > 0, f"understeer K {K:.5f} <= 0 (check ECC-12)"),
    ]
    return [msg for ok, msg in checks if not ok]


def invalid_reason(
    seg: Segment,
    v_target: float,
    window: int = 20,
    min_speed_frac: float = 0.85,
    steady: bool = True,
    max_r_std: float = 0.02,
) -> str | None:
    """Why a segment must not enter the fit or the held-out set (None when valid): it hit something, or ended
    (mean of the last `window` samples) below `min_speed_frac` x the target speed (stalled, slid or braked off).
    With `steady` (steps, not chirps) it must also hold a steady yaw rate: final-window std <= `max_r_std` rad/s
    (a larger one means a disturbance such as kerb contact, which is not registered as a collision)."""
    if seg.collided:
        return "collided"
    v = float(np.mean(seg.vx[-window:]))
    if v < min_speed_frac * v_target:
        return f"final speed {v:.1f} < {min_speed_frac} x {v_target:g} m/s"
    r_std = float(np.std(seg.r[-window:]))
    if steady and r_std > max_r_std:
        return f"not steady: r std {r_std:.3f} > {max_r_std:g} rad/s"
    return None


def steady_gain_error(seg: Segment, p: VehicleParams, K: float, window: int = 20) -> float:
    """|measured / predicted - 1| of the yaw-rate gain over the last `window` samples, predicted v / (L + K v^2)
    at the speed actually driven (the speed hold can miss its nominal speed)."""
    v = float(np.mean(seg.vx[-window:]))
    return abs(np.mean(seg.r[-window:]) / np.mean(seg.delta[-window:]) / (v / (p.L + K * v**2)) - 1.0)


def fit_longitudinal(s: LongSamples, p: VehicleParams) -> LongitudinalMap:
    """calibration.md §2 step 4: drive side linear over throttle 0.2-0.8 and 3-20 m/s, brake side through the origin."""
    need = s.ax + p.rolling + p.drag_coeff * s.vx**2  # drive accel required beyond the resistances
    drive = (s.vx >= 3.0) & (s.vx <= 20.0) & (s.throttle >= 0.2) & (s.throttle <= 0.8)
    gain, bias = np.polyfit(need[drive], s.throttle[drive], 1)
    braking = s.brake > 0.0
    req = -need[braking]
    brake_gain = float(np.sum(s.brake[braking] * req) / np.sum(req * req))
    coast = (s.throttle == 0.0) & (s.brake == 0.0) & (s.vx > 5.0)
    default = LongitudinalMap()
    band = float(np.max(np.abs(s.ax[coast]))) if coast.any() else default.coast_band
    return LongitudinalMap(float(gain), float(bias), brake_gain, band)


def throttle_r2(s: LongSamples, p: VehicleParams, lm: LongitudinalMap) -> float:
    """R^2 of the drive-side fit; below 0.9 the map should become a 2-D table (calibration.md §2, a code change)."""
    need = s.ax + p.rolling + p.drag_coeff * s.vx**2
    drive = (s.vx >= 3.0) & (s.vx <= 20.0) & (s.throttle >= 0.2) & (s.throttle <= 0.8)
    y, pred = s.throttle[drive], lm.throttle_bias + lm.throttle_gain * need[drive]
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2))


def fit_coastdown(vx: np.ndarray, ax: np.ndarray) -> tuple[float, float]:
    """(rolling, drag_coeff) from a = -rolling - drag_coeff vx^2 with throttle = brake = 0."""
    A = np.column_stack([np.ones_like(vx), vx**2])
    (rolling, drag), *_ = np.linalg.lstsq(A, -ax, rcond=None)
    return max(float(rolling), 0.0), max(float(drag), 0.0)


# --- friction --------------------------------------------------------------------------------------------------
def a_lat_limit(
    a_lat: np.ndarray, lateral: np.ndarray, slip: np.ndarray, dt: float, hold: float = 2.0,
    max_lateral: float = 1.0, max_slip: float = 0.1,
) -> float:  # fmt: skip
    """Highest a_lat sustained for `hold` s before the lateral deviation exceeds 1 m or |vy|/vx exceeds 0.1."""
    out = np.flatnonzero((np.abs(lateral) > max_lateral) | (np.abs(slip) > max_slip))
    end = int(out[0]) if len(out) else len(a_lat)
    w = round(hold / dt)
    if end < w:
        return 0.0
    return float(np.max(np.lib.stride_tricks.sliding_window_view(a_lat[:end], w).min(axis=1)))


def friction_scale_for(a_lat_max: dict[float, float], target: float) -> float:
    """The tire-friction scale k* with a_lat_max(k*) = target (m/s^2), by linear interpolation of the measured curve."""
    ks = sorted(a_lat_max)
    a = [a_lat_max[k] for k in ks]
    if any(y1 <= y0 for y0, y1 in itertools.pairwise(a)):
        raise ValueError(f"a_lat_max is not monotone in the friction scale: {dict(zip(ks, a, strict=True))}")
    if not a[0] <= target <= a[-1]:
        raise ValueError(f"target {target:.2f} m/s^2 outside the measured range {a[0]:.2f}-{a[-1]:.2f}")
    return float(np.interp(target, a, ks))
