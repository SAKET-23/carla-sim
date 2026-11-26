#!/usr/bin/env bash
# Smoke 02: Vulkan sees the NVIDIA GPU as root and as carla (runbook.md section 4).
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/smoke-02-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
die() { echo "smoke 02 FAIL: $*" >&2; exit 1; }
cd /workspace/carla-sim

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
short=${gpu#NVIDIA }
echo "nvidia-smi GPU: $gpu"

check() { # label, summary text
  local who=$1 out=$2 devices software
  devices=$(echo "$out" | grep -c 'deviceName' || true)
  software=$(echo "$out" | grep 'deviceName' | grep -ciE 'llvmpipe|lavapipe' || true)
  if [ "$devices" -gt 0 ] && [ "$devices" -eq "$software" ]; then die "$who: only software devices (llvmpipe/lavapipe)"; fi
  echo "$out" | grep 'deviceName' | grep -qiF "$short" || die "$who: no deviceName matching '$short'"
  echo "$out" | grep -q 'driverID *= *DRIVER_ID_NVIDIA_PROPRIETARY' || die "$who: no DRIVER_ID_NVIDIA_PROPRIETARY"
  echo "$who: ok"
}

echo "--- manifests"
ls -l /etc/vulkan/icd.d /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d 2>&1 || true
ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia' || true

root_out=$(vulkaninfo --summary 2>&1) || { echo "$root_out"; die "vulkaninfo failed as root"; }
echo "$root_out"
carla_out=$(runuser -u carla -- env NVIDIA_DRIVER_CAPABILITIES=all vulkaninfo --summary 2>&1) || { echo "$carla_out"; die "vulkaninfo failed as carla"; }
echo "$carla_out"
check root "$root_out"
check carla "$carla_out"
echo "smoke 02 PASS"
