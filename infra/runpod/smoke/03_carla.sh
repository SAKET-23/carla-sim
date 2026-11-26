#!/usr/bin/env bash
# Smoke 03: CARLA starts headless on the GPU (runbook.md section 4). Leaves CARLA and the VRAM logger running
# for 04-06; run_all.sh stops both on exit.
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/smoke-03-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
die() { echo "smoke 03 FAIL: $*" >&2; exit 1; }
cd /workspace/carla-sim

gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1; }
before=$(gpu_mib)
start_carla.sh --quality Low || die "start_carla.sh failed"

# VRAM logger (vram-budget.md section 3); run_all.sh kills it through the pid file
stamp=$(date -u +%Y%m%dT%H%M%SZ)
nohup nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
  --format=csv,nounits -l 1 > "/workspace/logs/gpu-$stamp.csv" 2>&1 &
echo $! > /workspace/logs/vram-logger.pid
nohup nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,nounits -l 5 \
  > "/workspace/logs/gpu-apps-$stamp.csv" 2>&1 &
echo $! >> /workspace/logs/vram-logger.pid

rose=0
for _ in $(seq 15); do
  now=$(gpu_mib)
  if [ $((now - before)) -ge 1024 ]; then rose=1; break; fi
  sleep 2
done
echo "memory.used: before ${before} MiB, now $(gpu_mib) MiB"
[ "$rose" = 1 ] || die "GPU memory.used did not rise by 1 GB within 30 s"
latest=$(compgen -G '/workspace/logs/carla-2*.log' | sort | tail -n 1)
if grep -E 'llvmpipe|lavapipe' "$latest"; then die "software rendering in $latest"; fi
echo "smoke 03 PASS"
