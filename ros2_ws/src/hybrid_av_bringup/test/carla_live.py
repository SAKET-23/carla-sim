"""Helpers for the live CARLA ROS tests (pod only): launch a graph and record topics. Not a test module."""

import json
import subprocess
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
STACK = ["ros2", "launch", "hybrid_av_bringup", "stack.launch.py", "sim:=carla", "foxglove:=false"]


def need_route(family: str = "nominal") -> None:
    table = yaml.safe_load((ROOT / "configs/scenarios/carla_routes.yaml").read_text()) or {}
    if not any(k == family or k.startswith(f"{family}_") for k in table):
        pytest.skip(f"no CARLA route for {family}: fill carla_routes.yaml (G2)")


def launch(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen([*STACK, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def spin_until_exit(node, proc: subprocess.Popen, timeout: float = 1200.0) -> str:
    import rclpy

    end = time.monotonic() + timeout
    while time.monotonic() < end and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
    if proc.poll() is None:
        proc.terminate()
    return proc.communicate(timeout=60)[0]


def stamp_ns(m) -> int:
    return m.header.stamp.sec * 10**9 + m.header.stamp.nanosec


def result(path: Path) -> dict:
    return json.loads(path.read_text())
