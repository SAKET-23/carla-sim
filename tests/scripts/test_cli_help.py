import subprocess
import sys

import pytest

SCRIPTS = [
    "scripts/bench_mpc.py",
    "scripts/run_scenario.py",
    "scripts/train_ppo.py",
    "scripts/eval_policy.py",
    "scripts/export_models.py",
    "scripts/identify_vehicle.py",
    "scripts/calibrate_friction.py",
    "scripts/find_map_features.py",
    "scripts/export_bundle.py",
    "scripts/run_edit_loop.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help(script):
    assert (
        subprocess.run([sys.executable, script, "--help"], check=False, capture_output=True).returncode == 0
    )
