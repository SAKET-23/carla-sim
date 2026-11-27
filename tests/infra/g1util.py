"""Shared helpers for the G1 pod-script tests: stub executables on PATH that record argv."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
G1 = ROOT / "infra" / "runpod" / "g1"

# Skipping for an old bash is allowed outside GitHub Actions only; under Actions (CI=true) a skip never happens (the test fails).
_old_bash = subprocess.run(["bash", "-c", "(( BASH_VERSINFO[0] >= 4 ))"], check=False).returncode != 0
need_bash4 = pytest.mark.skipif(_old_bash and not os.environ.get("GITHUB_ACTIONS"), reason="needs bash >= 4")


def stub(bindir: Path, name: str, body: str = "") -> Path:
    """Write an executable stub that records its argv to <bindir>/<name>.argv, then runs body."""
    bindir.mkdir(parents=True, exist_ok=True)
    p = bindir / name
    p.write_text(f'#!/bin/bash\nprintf \'%s\\n\' "$*" >> "{bindir}/{name}.argv"\n{body}\n')
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def calls(bindir: Path, name: str) -> list[str]:
    f = bindir / f"{name}.argv"
    return f.read_text().splitlines() if f.exists() else []


def run(script: Path, args: list[str], env: dict, bindir: Path, **kw):
    e = {k: v for k, v in os.environ.items() if k not in ("RUNPOD_API_KEY",)}
    e["PATH"] = f"{bindir}:{e['PATH']}"
    e["G1_PY"] = sys.executable
    e.update(env)
    return subprocess.run([str(script), *args], env=e, capture_output=True, text=True, check=False, **kw)
