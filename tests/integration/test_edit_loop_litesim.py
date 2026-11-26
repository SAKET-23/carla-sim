import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow


def test_bundle_to_loop_on_litesim(tmp_path, seeded_compact_policy_path):
    r = subprocess.run(
        [
            sys.executable,
            "scripts/export_bundle.py",
            "--policy",
            str(seeded_compact_policy_path),
            "--scenario",
            "wet_curve",
            "--out",
            str(tmp_path / "b"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode in (0, 2), r.stderr
    if r.returncode == 2:
        pytest.skip("seeded random policy produced no wet-curve failures")
    bundle = next((tmp_path / "b").glob("bundle_wet_curve_*"))
    r = subprocess.run(
        [
            sys.executable,
            "scripts/run_edit_loop.py",
            "--bundle",
            str(bundle),
            "--editor",
            "scripted",
            "--out",
            str(tmp_path / "e"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "e/summary.json").read_text())
    assert s["stop_reason"] in {"accepted", "max_iterations"}
    assert (tmp_path / "e/attempts/iter_1/gates.json").exists()
