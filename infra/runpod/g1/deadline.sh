#!/usr/bin/env bash
# On-pod deadline watchdog: flags at T-30 min and T-15 min (with a checkpoint), stop-pod call at T.
# Independent of the Mac watcher, so the cap holds if the Mac sleeps. G1_ONCE=1 (test hook): one tick, one stop attempt.
set -euo pipefail

[ -n "${POD_DEADLINE_EPOCH:-}" ] || { echo "deadline: POD_DEADLINE_EPOCH is required" >&2; exit 2; }
G1_DIR=${G1_DIR:-/workspace/g1}
TICK=${G1_TICK_S:-60}
NOW_CMD=${G1_NOW_CMD:-date +%s}
RUNNER=${G1_RUNNER:-$(dirname "$0")/run_g1.sh}
mkdir -p "$G1_DIR/flags" "$G1_DIR/logs"
LOG=$G1_DIR/logs/deadline.log
log() { echo "$(date -u +%FT%TZ) deadline: $*" | tee -a "$LOG"; }

if [ -z "${RUNPOD_API_KEY:-}" ]; then
  log "NO POD KEY: Mac watcher is the only stop path"
  touch "$G1_DIR/flags/no_pod_key"
fi

stop_pod() {
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" -d '{"action":"stop"}' "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID:-}/action") || code=000
  log "stop call returned HTTP $code"
  [ "$code" = 200 ]
}

while :; do
  rem=$((POD_DEADLINE_EPOCH - $(eval "$NOW_CMD")))
  if [ "$rem" -le 1800 ] && [ ! -e "$G1_DIR/flags/warn_30min" ]; then
    touch "$G1_DIR/flags/warn_30min"; log "30 min left"
  fi
  if [ "$rem" -le 900 ] && [ ! -e "$G1_DIR/flags/final_checkpoint" ]; then
    touch "$G1_DIR/flags/final_checkpoint"; log "15 min left: final checkpoint"
    "$RUNNER" --checkpoint || log "checkpoint failed"
  fi
  if [ "$rem" -le 0 ]; then
    [ -n "${RUNPOD_API_KEY:-}" ] || { log "deadline reached, cannot stop (no key)"; exit 0; }
    until stop_pod; do
      [ -z "${G1_ONCE:-}" ] || exit 1
      sleep "${G1_RETRY_S:-60}"
    done
    exit 0
  fi
  [ -z "${G1_ONCE:-}" ] || exit 0
  sleep "$TICK"
done
