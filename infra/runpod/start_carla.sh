#!/usr/bin/env bash
# Start a headless CARLA server as user carla (image-and-bootstrap.md section 6).
# Usage: start_carla.sh [--quality Low|Epic] [--port 2000] [--town TownXX]
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/start_carla-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
die() { echo "start_carla: $*" >&2; exit 1; }

# ECC-26: Low is the default quality (CARLA_QUALITY on the template); datasets and evaluation must use the same level
QUALITY="${CARLA_QUALITY:-Low}"
PORT=2000
TOWN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quality) QUALITY=${2:?--quality needs a value}; shift 2 ;;
    --port) PORT=${2:?--port needs a value}; shift 2 ;;
    --town) TOWN=${2:?--town needs a value}; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
case "$QUALITY" in Low|Epic) ;; *) die "quality must be Low or Epic, got $QUALITY" ;; esac
CARLA_ROOT="${CARLA_ROOT:-/workspace/carla/0.9.16}"
[ -x "$CARLA_ROOT/CarlaUE4.sh" ] || die "CARLA not installed in $CARLA_ROOT: run install_carla.sh"

# 1. per-port checks: the RPC port and the streaming port (PORT+1) must be free
for p in "$PORT" $((PORT + 1)); do
  if [ -n "$(ss -ltnH "sport = :$p")" ]; then die "port $p is in use (run stop_carla.sh --port $PORT)"; fi
done

# 2. launch (UE4 refuses root, so always runuser -u carla)
LOG="/workspace/logs/carla-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup runuser -u carla -- env NVIDIA_DRIVER_CAPABILITIES=all \
  "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port="$PORT" -quality-level="$QUALITY" \
  > "$LOG" 2>&1 &
echo "CARLA starting, log $LOG"

# 3. readiness: a Python client must connect within 120 s
READY=""
for _ in $(seq 60); do
  if READY=$(python -c "import carla; c=carla.Client('localhost',$PORT); c.set_timeout(5); w=c.get_world(); print(c.get_server_version(), w.get_map().name)" 2>/dev/null); then
    break
  fi
  READY=""
  sleep 2
done
if [ -z "$READY" ]; then
  tail -n 30 "$LOG" >&2 || true
  die "no client connection within 120 s (see $LOG)"
fi
echo "CARLA server: $READY"
case "$READY" in
  0.9.16*) ;;
  *) echo "WARNING: server version is not 0.9.16 (the tarballs were re-uploaded in Aug-Sep 2026): ECC-01" >&2 ;;
esac
# CarlaUE4.sh starts the shipping binary as a child, so $! is not the server's PID
pgrep -u carla -f "CarlaUE4-Linux-Shipping.*-carla-rpc-port=$PORT" | head -n 1 > "/workspace/logs/carla-$PORT.pid" || true
echo "$QUALITY" > "/workspace/logs/carla-$PORT.quality"

# 4. a software-rendered CARLA is useless even if a client connected
if grep -E 'llvmpipe|lavapipe|Vulkan.*failed' "$LOG"; then
  die "software rendering or Vulkan failure in $LOG"
fi

# 5. optional map
if [ -n "$TOWN" ]; then
  python -c "import carla; c=carla.Client('localhost',$PORT); c.set_timeout(120); c.load_world('$TOWN'); print('loaded $TOWN')"
fi
echo "CARLA ready on port $PORT (quality $QUALITY)"
