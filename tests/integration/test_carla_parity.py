"""IC-33: in-process run_episode(CarlaSim) vs the ROS graph on CARLA (integration-check.md section G).

Needs the ROS 2 environment on the pod (rclpy, `ros2 launch`) next to the CARLA server."""

import dataclasses
import subprocess
import time

import numpy as np
import pytest

from hybrid_av.eval.runner import run_episode
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.expert import ConstantAction
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.carla_backend.client import CarlaSimConfig
from hybrid_av.sim.carla_backend.routes import carla_scenario
from hybrid_av.sim.carla_backend.sim import CarlaSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from tests.integration.helpers import ROUTES

pytestmark = pytest.mark.carla
SEED, N = 100000, 100


def _record_ros(tmp_path):
    import rclpy
    from hybrid_av_msgs.msg import VehicleCommand
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    rclpy.init()
    node = rclpy.create_node("carla_parity_recorder")
    cmds, odom = {}, {}
    stamp = lambda m: m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1000)
    node.create_subscription(
        VehicleCommand,
        "/carla/ego/command",
        lambda m: cmds.__setitem__(stamp(m), (m.steer, m.throttle, m.brake)),
        qos,
    )
    node.create_subscription(
        Odometry,
        "/carla/ego/ground_truth/odom",
        lambda m: odom.__setitem__(stamp(m), (m.pose.pose.position.x, m.pose.pose.position.y)),
        qos,
    )
    proc = subprocess.Popen(
        ["ros2", "launch", "hybrid_av_bringup", "stack.launch.py", "sim:=carla", "mode:=ground_truth",
         "policy:=constant", "foxglove:=false", "scenario:=nominal", f"seed:={SEED}", f"max_ticks:={N}",
         f"result_path:={tmp_path / 'result.json'}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    end = time.monotonic() + 900
    while time.monotonic() < end and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
    out = proc.communicate(timeout=60)[0]
    node.destroy_node()
    rclpy.shutdown()
    assert proc.returncode == 0, out[-3000:]
    keys = sorted(cmds)
    return np.array([cmds[k] for k in keys]), np.array([odom[k] for k in sorted(odom)])


def test_parity_in_process_vs_ros(carla_server, tmp_path):
    """IC-33 — see Plan/06-integration/integration-check.md; likely fix: ECC-22."""
    pytest.importorskip("rclpy")
    try:
        sc = carla_scenario(make_scenario("nominal", SEED), "nominal", SEED, None, ROUTES)
    except LookupError:
        pytest.skip("no CARLA route for nominal: fill carla_routes.yaml (G2)")
    sc = dataclasses.replace(sc, time_limit=N * 0.05)  # the episode ends after N steps
    sim = CarlaSim(CarlaSimConfig(sensors=None, town=sc.carla_route["town"]))
    try:
        res = run_episode(
            sim,
            sc,
            AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig()),
            ConstantAction(),
        )
    finally:
        sim.close()
    ref = np.column_stack([res.trace[k] for k in ("steer", "throttle", "brake")])[:N]
    ref_xy = np.column_stack([res.trace["x"], res.trace["y"]])[:N]
    cmds, odom = _record_ros(tmp_path)
    assert len(cmds) >= N
    diff = np.abs(cmds[:N] - ref)
    assert diff.max() <= 1e-3, (
        f"first diverging step {int(np.argmax(diff.max(axis=1) > 1e-3))}; max {diff.max(axis=0)}"
    )
    assert np.hypot(*(odom[N - 1] - ref_xy[N - 1])) <= 0.05
