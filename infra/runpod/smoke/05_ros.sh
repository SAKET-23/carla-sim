#!/usr/bin/env bash
# Smoke 05: the ROS 2 bridge publishes at the expected rates (runbook.md section 4).
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/smoke-05-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
die() { echo "smoke 05 FAIL: $*" >&2; exit 1; }
cd /workspace/carla-sim

# ROS setup scripts read unset variables
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
[ -f ros2_ws/install/setup.bash ] && source ros2_ws/install/setup.bash
set -u

launch_log=/workspace/logs/smoke-05-launch.log
ros2 launch hybrid_av_bringup bridge_only.launch.py town:=Town03 > "$launch_log" 2>&1 &
LAUNCH_PID=$!
trap 'kill -INT "$LAUNCH_PID" 2> /dev/null || true; wait "$LAUNCH_PID" 2> /dev/null || true' EXIT

for _ in $(seq 120); do
  if timeout 5 ros2 topic echo --once /clock > /dev/null 2>&1; then ready=1; break; fi
  kill -0 "$LAUNCH_PID" 2> /dev/null || { tail -n 30 "$launch_log" >&2; die "bridge launch exited early"; }
  ready=0
done
[ "${ready:-0}" = 1 ] || { tail -n 30 "$launch_log" >&2; die "no /clock within the wait"; }
python /usr/local/share/carla-sim/smoke/bridge_rate_check.py --seconds 20 || die "bridge_rate_check.py failed"
echo "smoke 05 PASS"
