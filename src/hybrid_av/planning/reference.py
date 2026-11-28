"""Policy-modulated MPC reference; rules in Plan/01-theory/mpc-formulation.md section 7."""

from dataclasses import dataclass

import numpy as np

from hybrid_av.geometry import unwrap_near
from hybrid_av.mpc.controller import MPCConfig, Reference
from hybrid_av.planning.route import Route
from hybrid_av.types import EgoState, PolicyAction


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    a_lat_max: float = 4.5  # m/s², curvature speed cap (comfort) on dry road; deliberately above the wet limit 0.35 g = 3.43 m/s² (scenarios.md §2)
    a_lon_comfort: float = 2.0  # m/s², used for the stop-profile
    lookahead_time_min: float = 0.0
    delta_y_rate: float = 1.0  # m/s, max rate the lateral offset may move along the horizon
    stop_buffer: float = 2.0  # m before a stop point
    actuator_lag: float = 0.0  # s, drivetrain/brake response; the stop profile starts v * lag earlier


class ReferenceGenerator:
    def __init__(self, cfg: ReferenceConfig, mpc_cfg: MPCConfig) -> None:
        self.cfg, self.mpc_cfg = cfg, mpc_cfg

    def generate(
        self,
        route: Route,
        ego: EgoState,
        s0: float,
        action: PolicyAction,
        mu_est: float,
        stop_s: float | None = None,
    ) -> Reference:
        c, n, dt = self.cfg, self.mpc_cfg.N, self.mpc_cfg.dt
        a_lon = c.a_lon_comfort
        mu_est = max(mu_est, 0.0)
        a_lat = min(c.a_lat_max, self.mpc_cfg.friction_margin * mu_est * 9.81)

        # Rule 1: local cap, then backward-propagated over the look-ahead window (1 m grid).
        d_brake = ego.vx**2 / (2 * a_lon) + 10.0
        g = np.arange(0.0, d_brake + 1.0, 1.0)
        loc = route.sample(s0 + g)
        local = np.minimum(
            loc["speed_limit"] * action.alpha_v, np.sqrt(a_lat / np.maximum(np.abs(loc["kappa"]), 1e-4))
        )
        w = local**2 + 2 * a_lon * g
        cap_grid = np.sqrt(np.minimum.accumulate(w[::-1])[::-1] - 2 * a_lon * g)

        # Rule 3: stop point (a given stop_s at or behind s0 is ignored).
        stops = [stop_s] if stop_s is not None and stop_s > s0 else []
        if action.brake >= 0.5:
            stops.append(s0 + ego.vx**2 / (2 * abs(self.mpc_cfg.ax_min)))
        stop = min(stops) if stops else None

        # Rule 2 + 3: integrate in time; the stop cap may drop faster than comfort.
        s, v = np.empty(n + 1), np.empty(n + 1)
        s[0], vk = s0, ego.vx
        for k in range(n + 1):
            if k:
                vk += np.clip(np.interp(s[k - 1], s0 + g, cap_grid) - vk, -a_lon * dt, a_lon * dt)
                s[k] = s[k - 1] + v[k - 1] * dt
            if stop is not None:
                d_stop = max(stop - c.stop_buffer - s[k] - vk * c.actuator_lag, 0.0)
                vk = min(vk, np.sqrt(2 * a_lon * d_stop))
            v[k] = vk

        # Rule 4: lateral offset ramp, clipped to own lane plus one adjacent.
        pt = route.sample(s)
        lim = 1.5 * pt["lane_width"]
        d = np.empty(n + 1)
        dk = route.project(ego.x, ego.y, s_hint=s0).lateral
        for k in range(n + 1):
            if k:
                dk += np.clip(action.delta_y - dk, -c.delta_y_rate * dt, c.delta_y_rate * dt)
            dk = float(np.clip(dk, -lim[k], lim[k]))
            d[k] = dk

        # Rule 5: offset along the left normal; heading unwrapped near the ego.
        psi = pt["psi"] + (unwrap_near(pt["psi"][0], ego.psi) - pt["psi"][0])
        return Reference(
            x=pt["x"] - d * np.sin(pt["psi"]),
            y=pt["y"] + d * np.cos(pt["psi"]),
            psi=psi,
            v=v,
            kappa=pt["kappa"],
            s=s,
        )
