#!/bin/bash
# Mac-side G1 watcher: mirrors /workspace/g1 every interval, logs one status line, and hard-stops the pod at
# deadline-5min or when step 12_summary is done. Must run under /bin/bash 3.2 (no assoc arrays, mapfile or ${v,,}).
# Test hooks: G1_NOW_CMD, G1_ONCE=1 (one iteration), G1_POLL_S (stop-poll delay, default 10), G1_PY (python3).
# The RunPod key is read from the Keychain at stop time only, kept in a variable, never logged or put on a command line.
set -euo pipefail

# keep the Mac awake
if [ -z "${G1_CAFFEINATED:-}" ] && command -v caffeinate > /dev/null; then
  G1_CAFFEINATED=1 exec caffeinate -i "$0" "$@"
fi

POD="" HOST="" DEADLINE="" OUT="" INTERVAL=300 DRY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pod-id) POD=$2; shift 2 ;;
    --ssh) HOST=$2; shift 2 ;;
    --deadline) DEADLINE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --interval) INTERVAL=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "usage: mac_watch.sh --pod-id ID --ssh HOST --deadline EPOCH [--out DIR] [--interval S] [--dry-run]" >&2; exit 2 ;;
  esac
done
if [ -z "$POD" ] || [ -z "$HOST" ] || [ -z "$DEADLINE" ]; then
  echo "mac_watch: --pod-id, --ssh and --deadline are required" >&2
  exit 2
fi
OUT=${OUT:-$HOME/carla-g1/$POD}
PY=${G1_PY:-python3}
NOW_CMD=${G1_NOW_CMD:-date +%s}
API=https://api.runpod.io/v2/pods/$POD
mkdir -p "$OUT"

now() { eval "$NOW_CMD"; }
log() { echo "$*" | tee -a "$OUT/watch.log"; }
sync_g1() { rsync -az "$HOST:/workspace/g1/" "$OUT/"; }

# status_lines NOW: the interval line, then ALERT lines (stale status.json, no_pod_key)
status_lines() {
  "$PY" - "$OUT/status.json" "$1" "$DEADLINE" << 'PYEOF'
import calendar, json, sys, time
path, now, deadline = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
try:
    s = json.load(open(path))
except (OSError, ValueError):
    print(f"{stamp} ALERT no readable status.json yet")
    sys.exit(0)
g, r = s.get("gpu", {}), s.get("ram", {})
rem = max(deadline - now, 0)
flags = s.get("flags", [])
print(f"{stamp} step={s.get('current_step')} done={s.get('steps_done')} failed={s.get('steps_failed')} "
      f"remaining={rem // 3600}:{rem % 3600 // 60:02d} gpu={g.get('util_pct')}%/{g.get('mem_used_mib')}MiB "
      f"ram={r.get('used_mib')}MiB cost=${s.get('cost_usd_so_far')} flags={','.join(flags)}")
age = now - calendar.timegm(time.strptime(s["ts"], "%Y-%m-%dT%H:%M:%SZ"))
if age > 600:
    print(f"{stamp} ALERT status.json is {age // 60} min old")
if "no_pod_key" in flags:
    print(f"{stamp} ALERT no_pod_key: the on-pod watchdog cannot stop the pod")
PYEOF
}

summary_done() {
  "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["steps"]["12_summary"]["status"] != "done")' \
    "$OUT/state.json" 2> /dev/null
}

stop_pod() { # final sync (may fail), POST stop action, poll status until EXITED
  sync_g1 || log "ALERT final sync failed (continuing to stop)"
  if [ -n "$DRY" ]; then log "DRY-RUN: would stop $POD"; return 0; fi
  local key code body i=0
  key=$(security find-generic-password -s runpod-api -w) || { log "ALERT cannot read the runpod-api key from the Keychain"; return 1; }
  while [ "$i" -lt 30 ]; do
    code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST -H @- -H "Content-Type: application/json" -d '{"action":"stop"}' "$API/action" <<< "Authorization: Bearer $key") || code=000
    log "stop call returned HTTP $code"
    body=$(curl -sS -H @- "$API" <<< "Authorization: Bearer $key") || body=
    case "${body// /}" in *'"status":"EXITED"'*) log "STOPPED $POD"; return 0 ;; esac
    i=$((i + 1))
    sleep "${G1_POLL_S:-10}"
  done
  log "ALERT $POD not EXITED after $i stop attempts"
  return 1
}

while :; do
  t=$(now)
  if [ "$t" -ge $((DEADLINE - 300)) ]; then
    log "deadline reached: final sync and stop"
    stop_pod && exit 0
  else
    sync_g1 || log "ALERT rsync failed"
    status_lines "$t" | while IFS= read -r l; do log "$l"; done
    if summary_done; then
      log "12_summary done: final sync and stop"
      stop_pod && exit 0
    fi
  fi
  [ -z "${G1_ONCE:-}" ] || exit 0
  sleep "$INTERVAL"
done
