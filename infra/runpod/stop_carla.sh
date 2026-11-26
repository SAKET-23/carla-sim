#!/usr/bin/env bash
# Stop CARLA servers (image-and-bootstrap.md section 6). Usage: stop_carla.sh [--port 2000 | --all]
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/stop_carla-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
die() { echo "stop_carla: $*" >&2; exit 1; }

PORT=2000
ALL=0
case "${1:-}" in
  "") ;;
  --all) ALL=1 ;;
  --port) PORT=${2:?--port needs a value} ;;
  *) die "unknown argument: $1" ;;
esac

if [ "$ALL" = 1 ]; then
  PORTS=$(pgrep -u carla -fa "CarlaUE4-Linux-Shipping" | grep -oE -- '-carla-rpc-port=[0-9]+' | cut -d= -f2 || true)
  PATTERN="CarlaUE4-Linux-Shipping"
else
  PORTS=$PORT
  PATTERN="CarlaUE4-Linux-Shipping.*-carla-rpc-port=$PORT"
fi

# 1. restore asynchronous settings so a frozen synchronous server does not outlive its clients
for p in $PORTS; do
  python -c "
import carla
c = carla.Client('localhost', $p); c.set_timeout(5)
w = c.get_world(); s = w.get_settings()
s.synchronous_mode = False; s.fixed_delta_seconds = None
w.apply_settings(s)" 2>/dev/null || true
done

# 2. TERM, wait up to 20 s, then KILL (does not depend on the PID file)
pkill -TERM -u carla -f "$PATTERN" || true
for _ in $(seq 20); do
  pgrep -u carla -f "$PATTERN" > /dev/null || break
  sleep 1
done
if pgrep -u carla -f "$PATTERN" > /dev/null; then
  pkill -KILL -u carla -f "$PATTERN" || true
fi

# 3. bookkeeping files
if [ "$ALL" = 1 ]; then
  rm -f /workspace/logs/carla-*.pid /workspace/logs/carla-*.quality
else
  rm -f "/workspace/logs/carla-$PORT.pid" "/workspace/logs/carla-$PORT.quality"
fi
echo "CARLA stopped"
