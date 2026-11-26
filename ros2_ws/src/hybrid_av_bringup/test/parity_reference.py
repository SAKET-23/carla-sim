"""In-process reference loop for the ROS parity test. Pure: no rclpy.

Mirrors litesim_node + mpc_node + policy_node step for step, with the same float32 roundings."""

from pathlib import Path

import numpy as np
from hybrid_av_nodes.convert import roundtrip_action, roundtrip_observation, roundtrip_route

from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.policy.io import load_policy
from hybrid_av.policy.network import TorchPolicySource
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import ActuatorCommand
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams


def _f32(v: float) -> float:
    return float(np.float32(v))


def reference_commands(scenario: str, seed: int, policy_path: Path, n: int = 200) -> np.ndarray:
    """(n, 3) steer, throttle, brake that the ROS loop must reproduce."""
    params, lm, mpc_cfg = VehicleParams(), LongitudinalMap(), MPCConfig()
    sim = LiteSim(params, lm, mpc_cfg.dt)
    obs = sim.reset(make_scenario(scenario, seed))  # litesim_node publishes it at tick 0 without stepping
    stack = AutonomyStack(params, lm, mpc_cfg, ReferenceConfig())
    stack.reset(roundtrip_route(sim.route()))
    source = TorchPolicySource(load_policy(Path(policy_path))[0])
    cmd = ActuatorCommand(0.0, 0.0, 1.0)
    out, action = np.zeros((n, 3)), None
    for k in range(n):
        if k:
            obs, _ = sim.step(cmd)
        obs = roundtrip_observation(obs)
        frame = stack.features(obs)
        if k % 2 == 0:
            action = roundtrip_action(source(frame.z, frame))
        c = stack.act(obs, action, frame).command
        cmd = ActuatorCommand(_f32(c.steer), _f32(c.throttle), _f32(c.brake))
        out[k] = (cmd.steer, cmd.throttle, cmd.brake)
    return out
