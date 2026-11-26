import re
import shutil
import subprocess
from pathlib import Path

import pytest

SH = sorted(Path("infra").rglob("*.sh"))


def test_all_scripts_strict_mode():
    for p in SH:
        assert "set -euo pipefail" in p.read_text(), p


def test_carla_never_as_root():
    s = Path("infra/runpod/start_carla.sh").read_text()
    assert "runuser -u carla --" in s and "-RenderOffScreen" in s and "-nosound" in s
    assert re.search(r"llvmpipe|lavapipe", s)


def test_dockerfile_rules():
    d = Path("infra/runpod/Dockerfile").read_text()
    assert "ENV NVIDIA_DRIVER_CAPABILITIES=all" in d and "libnvidia-gl" not in d
    assert "useradd -m -u 1000" in d and "CARLA_0.9.16.tar.gz" not in d  # CARLA is not baked in
    assert re.search(r"^FROM nvidia/cuda:12\.\S+-cudnn-devel-ubuntu22\.04\s*$", d, re.MULTILINE)
    assert re.search(
        r'"numpy[^"]*<2"', Path("pyproject.toml").read_text()
    )  # spec: "numpy<2"; the pin is >=1.26,<2


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
def test_shellcheck_and_hadolint():
    files = [str(p) for p in SH]
    assert (
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{Path.cwd()}:/mnt",
                "-w",
                "/mnt",
                "koalaman/shellcheck:stable",
                *files,
            ],
            check=False,
        ).returncode
        == 0
    )
    conf = f"{Path.cwd()}/.hadolint.yaml:/.config/hadolint.yaml"
    with open("infra/runpod/Dockerfile") as dockerfile:
        cmd = ["docker", "run", "--rm", "-i", "-v", conf, "hadolint/hadolint"]
        assert subprocess.run(cmd, stdin=dockerfile, check=False).returncode == 0
