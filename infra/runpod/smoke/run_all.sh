#!/usr/bin/env bash
# Run smoke tests 01..NN in order, stopping at the first failure (runbook.md section 4).
# Usage: run_all.sh [--upto NN]   (default 06). Stops CARLA and the VRAM logger on exit, pass or fail.
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/smoke-run_all-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1

UPTO=6
case "${1:-}" in
  "") ;;
  --upto) UPTO=$((10#${2:?--upto needs a number})) ;;
  *) echo "run_all: unknown argument: $1" >&2; exit 1 ;;
esac
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
  if [ -f /workspace/logs/vram-logger.pid ]; then
    xargs -r kill < /workspace/logs/vram-logger.pid 2> /dev/null || true
    rm -f /workspace/logs/vram-logger.pid
  fi
  stop_carla.sh || true
}
trap cleanup EXIT

for i in $(seq 1 "$UPTO"); do
  n=$(printf '%02d' "$i")
  script=$(compgen -G "$DIR/${n}_*.sh" | head -n 1)
  [ -n "$script" ] || { echo "run_all: no script for $n" >&2; exit 1; }
  echo "=== $(basename "$script")"
  bash "$script" || { echo "run_all: $(basename "$script") failed" >&2; exit 1; }
done
echo "run_all: PASS up to $UPTO"
