"""Gymnasium environment over the autonomy stack (Plan/02-system-design/interfaces.md section 10)."""

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from hybrid_av.features.extractor import FeatureFrame
from hybrid_av.policy.network import scale_action
from hybrid_av.rl.reward import RewardConfig, compute_reward
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.base import Scenario, Sim
from hybrid_av.stack import AutonomyStack


class _InfoFrame(FeatureFrame):
    """FeatureFrame with array-safe equality, so gymnasium's determinism checks can compare info dicts."""

    def __eq__(self, other):
        return np.array_equal(self.raw, other.raw) and self.route_s == other.route_s


def nominal_sampler(rng: np.random.Generator) -> Scenario:
    return make_scenario("nominal", int(rng.integers(0, 10000)))


class DrivingEnv(gym.Env):
    def __init__(
        self,
        sim_factory: Callable[[], Sim],
        scenario_sampler: Callable[[np.random.Generator], Scenario],
        stack: AutonomyStack,
        reward_cfg: RewardConfig,
        policy_period: int = 2,
    ) -> None:
        self.sim, self.sampler, self.stack = sim_factory(), scenario_sampler, stack
        self.cfg, self.period = reward_cfg, policy_period
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (24,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._obs = self.sim.reset(self.sampler(self.np_random))
        self.route = self.sim.route()
        self.stack.reset(self.route)
        self._frame = self.stack.features(self._obs)
        self._prev = scale_action(np.zeros(3))
        return self._frame.z, {}

    def step(self, action):
        a = scale_action(np.clip(action, -1.0, 1.0))
        total = 0.0
        for k in range(self.period):
            out = self.stack.act(self._obs, a, self._frame)
            self._obs, info = self.sim.step(out.command)
            cur = self.stack.features(self._obs)
            r, _ = compute_reward(self._frame, cur, info, out.mpc, a, self._prev if k == 0 else a, self.cfg)
            total, self._frame = total + r, cur
            terminated = (
                info.collision != "none"
                or info.off_road
                or info.route_completed
                or info.route_deviation
                or info.blocked
            )
            if terminated or info.timeout:
                break
        self._prev = a
        info_out = {
            "step_info": info,
            "progress": min(1.0, info.route_s / self.route.length),
            "mpc_status": out.mpc.status,
            "frame": _InfoFrame(**vars(self._frame)),
        }
        return self._frame.z, total, terminated, info.timeout and not terminated, info_out
