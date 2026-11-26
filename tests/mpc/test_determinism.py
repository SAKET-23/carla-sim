import dataclasses

import numpy as np

from hybrid_av.config import load_dataclass
from hybrid_av.eval.runner import run_episode
from hybrid_av.mpc.controller import BicycleMPC, MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.expert import ConstantAction
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams


def test_default_has_no_wall_clock_limit():
    assert MPCConfig().time_limit == 0.0
    assert load_dataclass(MPCConfig, "configs/mpc/default.yaml") == MPCConfig()


def test_realtime_config_differs_only_in_time_limit():
    rt = load_dataclass(MPCConfig, "configs/mpc/realtime.yaml")
    assert rt.time_limit == 0.015
    assert dataclasses.replace(rt, time_limit=0.0) == MPCConfig()


def _trace(seed):
    sc, p, lm = make_scenario("wet_curve", seed), VehicleParams(), LongitudinalMap()
    sim = LiteSim(p, lm)
    r = run_episode(sim, sc, AutonomyStack(p, lm, MPCConfig(), ReferenceConfig()), ConstantAction())
    return np.stack([r.trace[k] for k in ("x", "y", "vx", "delta", "ax")])


def test_identical_closed_loop_runs_are_bit_identical():
    assert np.array_equal(_trace(50000), _trace(50000))


def test_reset_makes_mpc_equal_to_fresh():
    from hybrid_av.types import EgoState
    from tests.mpc.helpers import straight_reference

    def go(m):
        e = EgoState(0.0, 0.0, -0.5, 0.0, 10.0, 0.0, 0.0)
        r = m.solve(e, straight_reference(e, 12.0), mu=1.0)
        return r.control, r.predicted, r.cost

    used = BicycleMPC(VehicleParams(), MPCConfig())
    go(used)
    go(used)
    used.reset()
    a, b = go(used), go(BicycleMPC(VehicleParams(), MPCConfig()))
    assert a[0] == b[0] and a[2] == b[2] and np.array_equal(a[1], b[1])


def _short_trace(sim, stack, seed, steps=200):
    from hybrid_av.rl.expert import ExpertPolicy

    obs = sim.reset(make_scenario("nominal", seed))
    stack.reset(sim.route())
    src, rows = ExpertPolicy(), []
    for k in range(steps):
        f = stack.features(obs)
        if k % 2 == 0:
            action = src(f.z, f)
        out = stack.act(obs, action, f)
        rows.append([obs.ego.x, obs.ego.y, obs.ego.vx, out.control.delta, out.control.ax, *f.z])
        obs, info = sim.step(out.command)
        if info.route_completed or info.timeout:
            break
    return np.array(rows)


def test_reused_stack_is_seed_deterministic():
    p, lm = VehicleParams(), LongitudinalMap()
    sim, stack = LiteSim(p, lm), AutonomyStack(p, lm, MPCConfig(), ReferenceConfig())
    first = _short_trace(sim, stack, 50002)
    _short_trace(sim, stack, 50003)
    assert np.array_equal(first, _short_trace(sim, stack, 50002))
