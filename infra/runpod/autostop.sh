#!/usr/bin/env bash
# Auto-stop an idle pod (cost-and-hours.md section 3 rule 2). Run in tmux at login.
# Every 5 minutes: GPU < 5 %, no python/CarlaUE4/colcon above 10 % CPU, no SSH input for N minutes. After N = 30
# idle minutes, `runpodctl pod stop` (3.x syntax). `touch /tmp/autostop.hold` suppresses it (delete the file when the job ends).
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/autostop-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1

N=${AUTOSTOP_MINUTES:-30}
INTERVAL=${AUTOSTOP_INTERVAL_S:-300}

# ECC-29: runpodctl and RUNPOD_POD_ID are assumed to exist in the pod (G1 Task 3 Step 4)
command -v runpodctl > /dev/null || { echo "autostop: runpodctl not found" >&2; exit 1; }
[ -n "${RUNPOD_POD_ID:-}" ] || { echo "autostop: RUNPOD_POD_ID is not set" >&2; exit 1; }

busy() {
  [ -e /tmp/autostop.hold ] && return 0
  local util
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -n 1)
  [ "${util:-0}" -ge 5 ] && return 0
  ps -eo pcpu,comm | awk '$1 > 10 && ($2 ~ /python|CarlaUE4|colcon/) {f=1} END {exit !f}' && return 0
  # `who -u` idle column: "." (under a minute), "old" (over a day) or HH:MM
  who -u | awk -v n="$N" '
    { idle = $5
      if (idle == ".") f = 1
      else if (idle ~ /^[0-9]+:[0-9]+$/) { split(idle, t, ":"); if (t[1] * 60 + t[2] < n) f = 1 } }
    END { exit !f }' && return 0
  return 1
}

idle=0
while true; do
  if busy; then idle=0; else idle=$((idle + INTERVAL)); fi
  if [ "$idle" -ge $((N * 60)) ]; then
    echo "autostop: idle for $N min, stopping pod $RUNPOD_POD_ID"
    runpodctl pod stop "$RUNPOD_POD_ID"
    exit 0
  fi
  sleep "$INTERVAL"
done
