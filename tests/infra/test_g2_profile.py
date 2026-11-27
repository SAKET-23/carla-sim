import subprocess
import sys

import yaml

from tests.infra.g1util import ROOT, run

G2 = ROOT / "infra" / "runpod" / "g2"
RIG = ROOT / "configs" / "sensors" / "carla_rig.yaml"
ALL = ["A_norender", "none", "cameras", "cameras_lidar", "full", "full_lidar_light"]
HEADER = "profile,tick_hz,carla_cpu_cores,bridge_cpu_cores,pod_cpu_cores,throttled_pct,gpu_mem_peak_mib"


def _ladder(tmp_path, **env):
    r = run(
        G2 / "profile_tick.sh", [], {"OUT_DIR": str(tmp_path), "PROFILE_DRY": "1", **env}, tmp_path / "bin"
    )
    assert r.returncode == 0, r.stderr
    return (tmp_path / "tick_profile.csv").read_text().splitlines()


def _sensors(tmp_path, profile):
    out = tmp_path / f"{profile}.yaml"
    subprocess.run([sys.executable, str(G2 / "rig_subset.py"), str(RIG), profile, str(out)], check=True)
    return {s["id"]: s for s in yaml.safe_load(out.read_text())["sensors"]}


def test_dry_run_writes_all_rows(tmp_path):
    rows = _ladder(tmp_path)
    assert rows[0] == HEADER and [r.split(",")[0] for r in rows[1:]] == ALL
    assert all(len(r.split(",")) == 7 for r in rows)


def test_profiles_subset(tmp_path):
    rows = _ladder(tmp_path, PROFILES="A_norender cameras")
    assert [r.split(",")[0] for r in rows[1:]] == ["A_norender", "cameras"]


def test_rig_subsets(tmp_path):
    assert set(_sensors(tmp_path, "cameras")) == {"camera_front", "camera_front_semantic", "imu"}
    assert set(_sensors(tmp_path, "cameras_lidar")) == {
        "camera_front",
        "camera_front_semantic",
        "imu",
        "lidar",
    }
    assert len(_sensors(tmp_path, "full")) == 8
    assert _sensors(tmp_path, "full")["lidar"]["attributes"]["channels"] == "64"
    light = _sensors(tmp_path, "full_lidar_light")
    assert len(light) == 8
    assert light["lidar"]["attributes"]["channels"] == "32"
    assert light["lidar"]["attributes"]["points_per_second"] == "600000"
