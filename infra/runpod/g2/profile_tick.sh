#!/usr/bin/env bash
# G2: which part of the sim is CPU-bound? Runs a ladder of sensor profiles on Town03_Opt / Low and records, per profile,
# the /clock rate, mean CPU of CARLA and the bridge, pod cgroup CPU and throttling, and peak VRAM.
# Writes $OUT_DIR/tick_profile.csv; a failing profile gets `error=<reason>` in tick_hz and the ladder continues.
# Env: OUT_DIR (/workspace/g2), PROFILE_WINDOW_S (150), PROFILE_WARMUP_S (30), PROFILES (subset), G2_REPO,
# PROFILE_DRY=1 (fake numbers, no CARLA/ROS).
set -euo pipefail

OUT_DIR=${OUT_DIR:-/workspace/g2}
WINDOW=${PROFILE_WINDOW_S:-150}
WARMUP=${PROFILE_WARMUP_S:-30}
REPO=${G2_REPO:-/workspace/carla-sim}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CSV=$OUT_DIR/tick_profile.csv
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/rigs"
echo "profile,tick_hz,carla_cpu_cores,bridge_cpu_cores,pod_cpu_cores,throttled_pct,gpu_mem_peak_mib" > "$CSV"

cpu_ticks() { sed 's/.*) //' "/proc/$1/stat" | awk '{print $12 + $13}'; } # PID: utime+stime in clock ticks
cg_usage_us() { # cumulative cgroup CPU usec
  if [ -r /sys/fs/cgroup/cpu.stat ]; then awk '$1 == "usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat
  else echo $(($(cat /sys/fs/cgroup/cpuacct/cpuacct.usage) / 1000)); fi
}
cg_throttled_us() { # cumulative cgroup throttled usec, empty if unavailable
  if [ -r /sys/fs/cgroup/cpu.stat ]; then awk '$1 == "throttled_usec" {print $2}' /sys/fs/cgroup/cpu.stat
  elif [ -r /sys/fs/cgroup/cpu/cpu.stat ]; then awk '$1 == "throttled_time" {print int($2 / 1000)}' /sys/fs/cgroup/cpu/cpu.stat; fi
}
carla_pid() { pgrep -n -f CarlaUE4-Linux-Shipping || true; }

wait_ports_free() {
  for _ in $(seq 60); do
    ss -ltnH | grep -qE ':(2000|2001)\s' || return 0
    sleep 1
  done
  echo "ports 2000/2001 still in use after 60s" >&2
  return 1
}

# measure PROFILE: prints "tick_hz,carla_cores,bridge_cores,pod_cores,throttled_pct,gpu_peak"; nonzero exit with reason on stderr
measure() {
  local p=$1 rig=none nr=false hz=$OUT_DIR/logs/hz-$1.txt gpu=0 v launch hzpid cpid bpid
  case $p in
    A_norender) nr=true ;;
    none) ;;
    *) rig=$OUT_DIR/rigs/$p.yaml
       if [ "$p" = full ]; then cp "$REPO/configs/sensors/carla_rig.yaml" "$rig"
       else python3 "$HERE/rig_subset.py" "$REPO/configs/sensors/carla_rig.yaml" "$p" "$rig"; fi ;;
  esac
  stop_carla.sh > /dev/null 2>&1 || true
  wait_ports_free || return 1
  start_carla.sh --quality Low --town Town03_Opt > "$OUT_DIR/logs/carla-$p.log" 2>&1 || { echo "start_carla failed" >&2; return 1; }
  ros2 launch hybrid_av_bringup bridge_only.launch.py town:=Town03_Opt "rig:=$rig" "no_rendering:=$nr" > "$OUT_DIR/logs/bridge-$p.log" 2>&1 &
  launch=$!
  sleep "$WARMUP"
  cpid=$(carla_pid)
  bpid=$(pgrep -n -f carla_bridge || true)
  [ -n "$cpid" ] && [ -n "$bpid" ] || { kill "$launch" 2> /dev/null || true; echo "carla or bridge process not found" >&2; return 1; }
  local c0 b0 u0 t0 s0 s1 u1 c1 b1 t1 clk
  clk=$(getconf CLK_TCK)
  c0=$(cpu_ticks "$cpid"); b0=$(cpu_ticks "$bpid"); u0=$(cg_usage_us); t0=$(cg_throttled_us); s0=$(date +%s.%N)
  PYTHONUNBUFFERED=1 timeout "$WINDOW" ros2 topic hz /clock --window 1000 > "$hz" 2>&1 &
  hzpid=$!
  while kill -0 "$hzpid" 2> /dev/null; do
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1)
    [ "$v" -le "$gpu" ] || gpu=$v
    sleep 2
  done
  c1=$(cpu_ticks "$cpid"); b1=$(cpu_ticks "$bpid"); u1=$(cg_usage_us); t1=$(cg_throttled_us); s1=$(date +%s.%N)
  kill "$launch" 2> /dev/null || true
  wait "$launch" 2> /dev/null || true
  local rate
  rate=$(grep -o 'average rate: [0-9.]*' "$hz" | tail -n 1 | cut -d' ' -f3)
  [ -n "$rate" ] || { echo "no /clock rate (see $hz)" >&2; return 1; }
  awk -v rate="$rate" -v clk="$clk" -v c0="$c0" -v c1="$c1" -v b0="$b0" -v b1="$b1" -v u0="$u0" -v u1="$u1" \
    -v t0="${t0:-}" -v t1="${t1:-}" -v s0="$s0" -v s1="$s1" -v gpu="$gpu" 'BEGIN {
      e = s1 - s0
      thr = (t0 == "" || t1 == "") ? "" : sprintf("%.1f", (t1 - t0) / (e * 1e6) * 100)
      printf "%s,%.2f,%.2f,%.2f,%s,%s\n", rate, (c1 - c0) / clk / e, (b1 - b0) / clk / e, (u1 - u0) / 1e6 / e, thr, gpu }'
}

for p in ${PROFILES:-A_norender none cameras cameras_lidar full full_lidar_light}; do
  if [ -n "${PROFILE_DRY:-}" ]; then echo "$p,20.0,4.00,0.50,6.00,0.0,1000" >> "$CSV"; continue; fi
  if out=$(measure "$p" 2> "$OUT_DIR/logs/measure-err.txt"); then
    echo "$p,$out" >> "$CSV"
  else
    echo "$p,error=$(tr ',\n' '; ' < "$OUT_DIR/logs/measure-err.txt"),,,,," >> "$CSV"
  fi
done
if [ -z "${PROFILE_DRY:-}" ]; then stop_carla.sh > /dev/null 2>&1 || true; fi
