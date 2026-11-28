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


def test_identify_drive_stops_at_a_collision():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace as NS

    spec = importlib.util.spec_from_file_location(
        "identify_vehicle", Path(__file__).resolve().parents[2] / "scripts" / "identify_vehicle.py"
    )
    iv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iv)

    class Sim:
        dt, n = 0.05, 0

        def reset(self, _):
            return NS(ego=NS(t=0.0, vx=10.0, vy=0.0, r=0.0))

        def step(self, _):
            self.n += 1
            return NS(ego=NS(t=self.n * 0.05, vx=10.0, vy=0.0, r=0.0)), NS(
                collision="static" if self.n == 7 else "none", route_completed=False
            )

    seg = iv.drive(Sim(), {}, 10.0, 5.0, lambda t: 0.0, 10.0, iv.VehicleParams(), iv.LongitudinalMap())
    assert seg.collided and len(seg.t) == 7
