#!/usr/bin/env bash
# Smoke 04: the Python API works and the camera renders (runbook.md section 4).
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/smoke-04-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
cd /workspace/carla-sim
python /usr/local/share/carla-sim/smoke/client_check.py || { echo "smoke 04 FAIL: client_check.py failed" >&2; exit 1; }
echo "smoke 04 PASS"
