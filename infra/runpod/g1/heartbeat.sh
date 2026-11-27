#!/usr/bin/env bash
# Heartbeat: every G1_HB_S seconds append a metrics row and rewrite status.json atomically.
# G1_ONCE=1 (test hook): one sample, then exit. G1_PY: python with json (default /opt/venv/bin/python).
set -euo pipefail

G1_DIR=${G1_DIR:-/workspace/g1}
: "${POD_DEADLINE_EPOCH:?POD_DEADLINE_EPOCH is required}"
PY=${G1_PY:-/opt/venv/bin/python}
NOW_CMD=${G1_NOW_CMD:-date +%s}
HEADER=ts,gpu_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,gpu_temp_c,gpu_power_w,ram_used_mib,ram_total_mib,load1,disk_used_gb,carla_running
mkdir -p "$G1_DIR/flags"

sample() {
  local now gpu ram load disk carla row
  now=$(eval "$NOW_CMD")
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' ') || gpu=
  [ -n "$gpu" ] || gpu=,,,,
  ram=$(free -m | awk '/^Mem:/ {print $3 "," $2}')
  load=$(cut -d' ' -f1 /proc/loadavg)
  disk=$(df -BG "$G1_DIR" | awk 'NR==2 {gsub("G", "", $3); print $3}')
  carla=0
  pgrep -f '[C]arlaUE4-Linux-Shipping' > /dev/null && carla=1
  row=$(date -u -d "@$now" +%FT%TZ),$gpu,$ram,$load,$disk,$carla
  [ -s "$G1_DIR/metrics.csv" ] || echo "$HEADER" > "$G1_DIR/metrics.csv"
  echo "$row" >> "$G1_DIR/metrics.csv"
  "$PY" - "$HEADER" "$row" "$now" "$POD_DEADLINE_EPOCH" "$G1_DIR" "${RUNPOD_POD_ID:-}" "$(hostname)" \
    > "$G1_DIR/status.json.tmp" << 'PYEOF'
import json, os, sys
header, row, now, deadline, g1, pod, host = sys.argv[1:]
m = dict(zip(header.split(","), row.split(",")))
now, deadline = int(now), int(deadline)
elapsed = now - (deadline - 36000)
try:
    steps = json.load(open(f"{g1}/state.json"))["steps"]
except (OSError, ValueError, KeyError):
    steps = {}
count = lambda s: sum(v.get("status") == s for v in steps.values())
cur = next((k for k, v in steps.items() if v.get("status") == "running"), None)
num = lambda k: float(m[k]) if m[k] else None
print(json.dumps({
    "ts": m["ts"], "pod_id": pod, "host": host,
    "elapsed_s": elapsed, "remaining_s": deadline - now,
    "current_step": cur, "steps_done": count("done"), "steps_failed": count("failed"),
    "cost_usd_so_far": round(elapsed / 3600 * 0.352, 4),
    "gpu": {"util_pct": num("gpu_util_pct"), "mem_used_mib": num("gpu_mem_used_mib"),
            "mem_total_mib": num("gpu_mem_total_mib"), "temp_c": num("gpu_temp_c"),
            "power_w": num("gpu_power_w")},
    "ram": {"used_mib": num("ram_used_mib"), "total_mib": num("ram_total_mib"), "load1": num("load1")},
    "disk": {"used_gb": num("disk_used_gb")},
    "carla_running": m["carla_running"] == "1",
    "flags": sorted(os.listdir(f"{g1}/flags")),
}))
PYEOF
  mv "$G1_DIR/status.json.tmp" "$G1_DIR/status.json"
}

while :; do
  sample
  [ -z "${G1_ONCE:-}" ] || exit 0
  sleep "${G1_HB_S:-30}"
done
