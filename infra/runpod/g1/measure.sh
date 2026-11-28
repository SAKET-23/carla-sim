#!/usr/bin/env bash
# G1 step 08: peak VRAM / RAM, mean CPU and /clock tick rate for {Town03_Opt, Town10HD} x {Low, Epic}, then the RL-mode row.
# Writes $G1_DIR/vram.csv (a row is replaced if its town,quality key already exists). A failing config gets
# `error=<reason>` in the gpu_mem_peak_mib column and the next one still runs.
# Test hooks: G1_MEASURE_DRY=1 (fake numbers, no CARLA), G1_REPO (default /workspace/carla-sim).
set -euo pipefail

G1_DIR=${G1_DIR:-/workspace/g1}
REPO=${G1_REPO:-/workspace/carla-sim}
CSV=$G1_DIR/vram.csv
HEADER=town,quality,gpu_mem_peak_mib,ram_peak_mib,cpu_mean_pct,tick_hz_mean,ticks
mkdir -p "$G1_DIR/logs"

put_row() { # TOWN QUALITY ROW: replace any row with the same town,quality key
  local key=$1,$2 row=$3 tmp
  [ -s "$CSV" ] || echo "$HEADER" > "$CSV"
  tmp=$(mktemp "$G1_DIR/.vram.XXXXXX")
  grep -v -F -x -e "$HEADER" "$CSV" | grep -v "^$key," > "$tmp" || true
  { echo "$HEADER"; cat "$tmp"; echo "$row"; } > "$CSV"
  rm -f "$tmp"
}

# measure TOWN QUALITY: prints "gpu_peak,ram_peak,cpu_mean,hz,ticks"; nonzero exit with the reason on stderr
measure() {
  local town=$1 q=$2 hz=$G1_DIR/logs/hz-$1-$2.txt gpu=0 ram=0 n=0 cpu_sum=0 v c u launch hzpid
  if [ -n "${G1_MEASURE_DRY:-}" ]; then echo "1000,2000,30.0,20.0,600"; return 0; fi
  stop_carla.sh || true
  start_carla.sh --quality "$q" --town "$town" || { echo "start_carla failed" >&2; return 1; }
  ros2 launch hybrid_av_bringup bridge_only.launch.py "town:=$town" "quality:=$q" > "$G1_DIR/logs/bridge-$town-$q.log" 2>&1 &
  launch=$!
  timeout "${G1_HZ_TIMEOUT_S:-600}" ros2 topic hz /clock --window 600 > "$hz" 2>&1 &
  hzpid=$!
  while kill -0 "$hzpid" 2> /dev/null; do
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1)
    [ "$v" -le "$gpu" ] || gpu=$v
    u=$(free -m | awk '/^Mem:/ {print $3}')
    [ "$u" -le "$ram" ] || ram=$u
    c=$(mpstat 1 1 | awk '/^Average/ {print 100 - $NF}')
    cpu_sum=$(awk -v a="$cpu_sum" -v b="$c" 'BEGIN {print a + b}')
    n=$((n + 1))
    sleep 2
  done
  kill "$launch" 2> /dev/null || true
  wait "$launch" 2> /dev/null || true
  local rate ticks
  rate=$(grep -o 'average rate: [0-9.]*' "$hz" | tail -n 1 | cut -d' ' -f3)
  ticks=$(grep -o 'window: [0-9]*' "$hz" | tail -n 1 | cut -d' ' -f2)
  [ -n "$rate" ] || { echo "no /clock rate (see $hz)" >&2; return 1; }
  echo "$gpu,$ram,$(awk -v s="$cpu_sum" -v n="$n" 'BEGIN {printf "%.1f", s / (n ? n : 1)}'),$rate,$ticks"
}

for town in Town03_Opt Town10HD; do
  for q in Low Epic; do
    if out=$(measure "$town" "$q" 2> "$G1_DIR/logs/measure-err.txt"); then
      put_row "$town" "$q" "$town,$q,$out"
    else
      put_row "$town" "$q" "$town,$q,error=$(tr ',\n' '; ' < "$G1_DIR/logs/measure-err.txt"),,,,"
    fi
  done
done
if [ -z "${G1_MEASURE_DRY:-}" ]; then stop_carla.sh || true; fi

# RL mode (IC-40): needs the flag in run_scenario.py, otherwise deferred to G2
if [ -z "${G1_MEASURE_DRY:-}" ] && grep -q -- --measure-tick-rate "$REPO/scripts/run_scenario.py" 2> /dev/null; then
  if out=$(cd "$REPO" && python scripts/run_scenario.py --backend carla --measure-tick-rate --mode A 2>&1 | tee "$G1_DIR/logs/rl_tick_rate.log" | tail -n 1); then
    put_row rl_mode_A - "rl_mode_A,-,,,,$(echo "$out" | grep -oE '[0-9]+(\.[0-9]+)?' | tail -n 1),"
  else
    put_row rl_mode_A - "rl_mode_A,-,error=run_scenario failed,,,,"
  fi
else
  put_row rl_mode_A - "rl_mode_A,-,deferred to G2 IC-40,,,,"
fi
