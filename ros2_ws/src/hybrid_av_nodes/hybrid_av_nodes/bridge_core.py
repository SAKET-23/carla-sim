"""Pure helpers of the CARLA bridge (simulation-backends.md section 2, interfaces.md section 14): no rclpy, no carla."""

import math
from collections.abc import Sequence

import numpy as np

from hybrid_av.eval.driving_score import driving_score
from hybrid_av.eval.runner import INFRACTION_KEYS
from hybrid_av.perception.base import RadarDetection
from hybrid_av.tracking.radar import RadarExtrinsic, detection_to_base


def image_bgra_to_rgb(raw: bytes, w: int, h: int) -> np.ndarray:
    return np.ascontiguousarray(np.frombuffer(raw, np.uint8).reshape(h, w, 4)[:, :, 2::-1])


def semantic_tags(raw: bytes, w: int, h: int) -> np.ndarray:
    # ECC-31: the semantic tag is the R channel of the BGRA buffer
    return np.ascontiguousarray(np.frombuffer(raw, np.uint8).reshape(h, w, 4)[:, :, 2])


def camera_info(w: int, h: int, fov_deg: float) -> dict:
    f = w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))  # square pixels, principal point at the centre
    cx, cy = w / 2.0, h / 2.0
    return {
        "width": w, "height": h, "distortion_model": "plumb_bob", "d": [0.0] * 5,
        "k": [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }  # fmt: skip


def radar_points(dets: Sequence[RadarDetection], ext: RadarExtrinsic) -> np.ndarray:
    """(N, 7) float32: x, y, z (base_link), velocity, azimuth, altitude, depth."""
    rows = [[*detection_to_base(d, ext), d.velocity, d.azimuth, d.altitude, d.depth] for d in dets]
    return np.array(rows, np.float32).reshape(-1, 7)


def imu_stamps(t_tick: float, n: int, dt_imu: float = 0.01) -> list[float]:
    # ECC-07: n samples per tick, the last one belongs to the tick time
    return [t_tick - (n - 1 - i) * dt_imu for i in range(n)]


class WheelOdomSynth:
    """Wheel odometry from ground-truth body velocity: N(0, sigma) noise and a +1 % scale (as litesim_node)."""

    def __init__(self, seed: int, sigma: float = 0.05, scale: float = 1.01) -> None:
        self.rng, self.sigma, self.scale = np.random.default_rng(seed), sigma, scale

    def __call__(self, vx: float, vy: float) -> tuple[float, float]:
        n = self.rng.normal(0.0, self.sigma, 2)
        return self.scale * vx + float(n[0]), self.scale * vy + float(n[1])


class EpisodeDriver:
    """Steps any `Sim` for the tick loop: tick 0 yields the reset observation without stepping, so command k
    answers observation k (same rule as litesim_node); accumulates infractions for the episode summary."""

    def __init__(self, sim, scenario) -> None:
        self.sim, self.dt = sim, sim.dt
        self.obs = sim.reset(scenario)
        self.route = sim.route()
        self.infractions = dict.fromkeys(INFRACTION_KEYS, 0)
        self.info = None
        self._first = True

    def step(self, cmd) -> tuple[float, bool]:
        if self._first:
            self._first = False
            return self.obs.ego.t, False
        self.obs, info = self.sim.step(cmd)
        self.info = info
        events = {"red_light": info.red_light_violation, "stop_sign": info.stop_sign_violation}
        if info.collision != "none":
            events[f"collision_{info.collision}"] = True
        for key, hit in events.items():
            self.infractions[key] += bool(hit)
        if info.off_road:
            self.infractions["off_road_m"] += self.obs.ego.vx * self.dt
        end = info.route_completed or info.route_deviation or info.blocked or info.timeout
        return self.obs.ego.t, bool(end or info.collision != "none")

    def summary(self, scenario: str, seed: int) -> dict:
        """EpisodeResult fields the bridge can know (the MPC statistics live in mpc_node)."""
        info = self.info
        done = info is not None and info.route_completed
        completion = 1.0 if done else min(1.0, (info.route_s if info else 0.0) / self.route.length)
        inf = dict(self.infractions)
        inf["timeout"] = int(bool(info and info.timeout and not info.route_completed))
        inf["blocked"] = int(bool(info and info.blocked))
        return {
            "scenario": scenario,
            "seed": seed,
            "route_completion": completion,
            "infractions": inf,
            "driving_score": driving_score(completion, inf, self.route.length),
            "duration": self.obs.ego.t,
        }
