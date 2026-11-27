#!/usr/bin/env bash
# Install CARLA 0.9.16 onto the network volume (image-and-bootstrap.md section 4). Idempotent.
# Usage: install_carla.sh [--with-additional-maps]
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/install_carla-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
die() { echo "install_carla: $*" >&2; exit 1; }

CARLA_ROOT="${CARLA_ROOT:-/workspace/carla/0.9.16}"
DL=/workspace/carla/downloads
BASE=https://downloads.carlasim.com/Linux
MAPS=0
case "${1:-}" in
  "") ;;
  --with-additional-maps) MAPS=1 ;;
  *) die "unknown argument: $1" ;;
esac

if [ -f "$CARLA_ROOT/CarlaUE4.sh" ] && [ "$(cat "$CARLA_ROOT/.installed" 2>/dev/null)" = "0.9.16" ] && [ "$MAPS" = 0 ]; then
  echo "CARLA 0.9.16 already installed in $CARLA_ROOT"
  exit 0
fi

# fetch <file name>: resumable download, size check against Content-Length, one retry
fetch() {
  local name=$1 url="$BASE/$1" f="$DL/$1" want have attempt
  want=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:" {v=$2} END {print (v == "" ? 0 : v)}')
  [ "$want" -gt 0 ] || die "no Content-Length from $url"
  for attempt in 1 2; do
    wget -c --progress=dot:giga -O "$f" "$url" || true
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$have" = "$want" ]; then return 0; fi
    echo "size mismatch for $name: $have of $want bytes (attempt $attempt)" >&2
    rm -f "$f"
  done
  die "download of $name failed the size check twice (HTTP URL $url)"
}

# CARLA publishes no checksum: record the first verified download's sha256 and compare on any re-download
verify_sha() {
  local name=$1 sumfile="$DL/${1%.tar.gz}.sha256" now
  now=$(sha256sum "$DL/$name" | awk '{print $1}')
  if [ -f "$sumfile" ]; then
    [ "$(cat "$sumfile")" = "$now" ] || die "$name sha256 differs from the recorded $sumfile"
  else
    echo "$now" > "$sumfile"
  fi
}

mkdir -p "$DL" "$CARLA_ROOT"
fetch CARLA_0.9.16.tar.gz
verify_sha CARLA_0.9.16.tar.gz
pv "$DL/CARLA_0.9.16.tar.gz" | tar --no-same-owner -xz -C "$CARLA_ROOT"
if [ "$MAPS" = 1 ]; then
  fetch AdditionalMaps_0.9.16.tar.gz
  pv "$DL/AdditionalMaps_0.9.16.tar.gz" | tar --no-same-owner -xz -C "$CARLA_ROOT"
  (cd "$CARLA_ROOT" && ./ImportAssets.sh)
fi
chown -R carla:carla "$CARLA_ROOT" 2> /dev/null || echo "install_carla: chown not supported on this volume (files are world-writable)"
if [ "${KEEP_TARBALL:-0}" != 1 ]; then rm -f "$DL"/CARLA_0.9.16.tar.gz "$DL"/AdditionalMaps_0.9.16.tar.gz; fi
echo 0.9.16 > "$CARLA_ROOT/.installed"

python -c "import carla; print(carla.__file__)"
# ECC-01: server tarball and the carla==0.9.16 wheel must be the same build. No wheel is installed here (/opt/venv is
# on the container disk and reset every pod start); if they differ, the fix goes into entrypoint.sh.
echo "installed carla wheel: $(python -c 'import importlib.metadata as m; print(m.version("carla"))')"
echo "files in PythonAPI/carla/dist:"
ls -1 "$CARLA_ROOT/PythonAPI/carla/dist/"
