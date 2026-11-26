import dataclasses

import numpy as np
from gymnasium.utils.env_checker import check_env

from hybrid_av.sim.litesim.sim import LiteSim
from tests.rl.helpers import _env


def test_check_env():
    check_env(_env(), skip_render_check=True)


def test_step_advances_two_sim_steps():
    e = _env()
    e.reset(seed=0)
    _, _, _, _, info = e.step(np.zeros(3, np.float32))
    t1 = info["step_info"].sim_time
    _, _, _, _, info = e.step(np.zeros(3, np.float32))
    assert abs(info["step_info"].sim_time - t1 - 0.10) < 1e-9


def test_reset_deterministic():
    a, _ = _env().reset(seed=3)
    b, _ = _env().reset(seed=3)
    np.testing.assert_array_equal(a, b)


calls = []


class CrashSim(LiteSim):  # reports a vehicle collision on every sim step
    def step(self, cmd):
        obs, info = super().step(cmd)
        calls.append(1)
        return obs, dataclasses.replace(info, collision="vehicle")


def test_termination_mid_period():
    e = _env(sim_cls=CrashSim)
    e.reset(seed=0)
    calls.clear()
    _, reward, terminated, truncated, _ = e.step(np.zeros(3, np.float32))
    assert terminated and not truncated and len(calls) == 1 and reward < 0


def test_timeout_is_truncation():  # time limit = 2 sim steps
    e = _env(edit=lambda sc: dataclasses.replace(sc, time_limit=0.1))
    e.reset(seed=0)
    _, _, terminated, truncated, info = e.step(np.zeros(3, np.float32))
    assert truncated and not terminated and info["step_info"].timeout
