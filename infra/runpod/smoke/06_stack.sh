#!/usr/bin/env bash
# Smoke 06: the full stack drives one route in ground-truth mode (runbook.md section 4).
# POLICY=<checkpoint path> after Phase G4 (default: constant).
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/smoke-06-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
die() { echo "smoke 06 FAIL: $*" >&2; exit 1; }
cd /workspace/carla-sim

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
[ -f ros2_ws/install/setup.bash ] && source ros2_ws/install/setup.bash
set -u

RESULT=/workspace/runs/smoke06/episode.json
mkdir -p "$(dirname "$RESULT")"
rm -f "$RESULT"
ros2 launch hybrid_av_bringup stack.launch.py sim:=carla mode:=ground_truth scenario:=nominal seed:=0 \
  "policy:=${POLICY:-constant}" foxglove:=false "result_path:=$RESULT" || die "stack launch failed"
[ -f "$RESULT" ] || die "no $RESULT written"

python - "$RESULT" <<'PY' || die "pass criteria not met"
import json, sys
r = json.load(open(sys.argv[1]))
inf = r["infractions"]
collisions = inf.get("collision_pedestrian", 0) + inf.get("collision_vehicle", 0) + inf.get("collision_static", 0)
print("route_completion", r["route_completion"], "collisions", collisions, "mpc_fallbacks", r.get("mpc_fallbacks", 0))
sys.exit(0 if r["route_completion"] >= 0.9 and collisions == 0 and r.get("mpc_fallbacks", 0) == 0 else 1)
PY
echo "smoke 06 PASS"
