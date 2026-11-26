import json
import subprocess
import sys

import numpy as np


def test_cli_writes_outputs(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "scripts/run_scenario.py",
            "--family",
            "wet_curve",
            "--seed",
            "50000",
            "--source",
            "constant",
            "--out",
            str(tmp_path),
        ],
        check=True,
    )
    run = next(tmp_path.iterdir())
    assert {"trace.npz", "metrics.json", "episode.png"} <= {p.name for p in run.iterdir()}
    assert "route_completion" in json.loads((run / "metrics.json").read_text())
    assert np.load(run / "trace.npz")["z"].shape[1] == 24


def test_cli_policy_source_not_yet_available():
    r = subprocess.run(
        [
            sys.executable,
            "scripts/run_scenario.py",
            "--family",
            "nominal",
            "--seed",
            "50000",
            "--source",
            "policy:x.pt",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "policy sources arrive in Phase 3" in r.stderr
