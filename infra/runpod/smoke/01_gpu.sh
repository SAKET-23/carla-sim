#!/usr/bin/env bash
# Smoke 01: the GPU is visible with graphics capabilities (runbook.md section 4).
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/smoke-01-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
die() { echo "smoke 01 FAIL: $*" >&2; exit 1; }
cd /workspace/carla-sim

rows=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits) || die "nvidia-smi failed"
echo "$rows"
[ "$(echo "$rows" | wc -l)" -eq 1 ] || die "expected exactly one GPU"
mem=$(echo "$rows" | awk -F', ' '{print $2}')
[ "$mem" -ge 23552 ] || die "GPU has $mem MiB, need >= 23 GB"
caps="${NVIDIA_DRIVER_CAPABILITIES:-}"
case "$caps" in all|*graphics*) ;; *) die "NVIDIA_DRIVER_CAPABILITIES is '$caps', need all (or graphics)" ;; esac
echo "driver: $(echo "$rows" | awk -F', ' '{print $3}')"
echo "smoke 01 PASS"
