"""bridge_only.launch.py and carla_bridge without a CARLA server (ROS dev container only)."""

import subprocess

import pytest

pytestmark = pytest.mark.ros
CODE = "import sys, hybrid_av_nodes.carla_bridge, hybrid_av_nodes.command_responder; assert 'carla' not in sys.modules"


def test_bridge_modules_import_without_carla():
    r = subprocess.run(
        ["python", "-c", CODE],
        check=False, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr


def test_bridge_only_declares_arguments():
    r = subprocess.run(
        ["ros2", "launch", "hybrid_av_bringup", "bridge_only.launch.py", "--show-args"],
        check=False, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert r.returncode == 0
    assert all(k in r.stdout for k in ("town", "scenario", "seed"))


def test_bridge_only_exits_nonzero_without_carla():
    """No server: the spawn-point fallback supplies the route, so the bridge fails at the CARLA connection."""
    r = subprocess.run(
        ["ros2", "launch", "hybrid_av_bringup", "bridge_only.launch.py"],
        check=False, capture_output=True, text=True, timeout=90,
    )  # fmt: skip
    out = r.stdout + r.stderr
    assert "carla_bridge" in out and ("bridge failed" in out or "Traceback" in out)
    assert "no CARLA route" not in out
