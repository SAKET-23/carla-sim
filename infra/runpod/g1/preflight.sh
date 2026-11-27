#!/usr/bin/env bash
# G1 step 00: host checks. Exit 20 = host GPU fault (replace the host), 1 = config problem, 0 = ok.
# Writes $G1_DIR/hostinfo.json either way. G1_WORKSPACE / G1_MIN_FREE_GB are test hooks.
# G1_EXPECT_GPU: substring the GPU name must contain (default 4090).
set -euo pipefail

G1_DIR=${G1_DIR:-/workspace/g1}
WS=${G1_WORKSPACE:-/workspace}
MIN_FREE=${G1_MIN_FREE_GB:-40}
mkdir -p "$G1_DIR"

# 1. a real CUDA op, before anything else
torch_out=$(python -c "import torch; x=torch.randn(512,512,device='cuda'); print(torch.__version__, torch.version.cuda, (x@x).sum().item())" 2>&1) || torch_out="FAIL $torch_out"
if ! smi=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1); then
  echo "preflight: nvidia-smi failed: $smi" >&2
  exit 1
fi
case "$torch_out" in
  FAIL*) echo "HOST GPU FAULT: ${torch_out#FAIL }" >&2; exit 20 ;;
esac
torch_cuda=$(echo "$torch_out" | tail -n 1 | cut -d' ' -f2)

# 2. facts and config checks
gpu=$(echo "$smi" | head -n 1 | cut -d, -f1 | sed 's/^ *//; s/ *$//')
driver=$(echo "$smi" | head -n 1 | cut -d, -f2 | tr -d ' ')
cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -n 1 | cut -d' ' -f3 || true)
vcpu=$(nproc)
ram_gb=$(awk '/^MemTotal:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
free_gb=$(df -BG --output=avail "$WS" | awk 'NR==2 {gsub("G", "", $1); print $1}')

fail=0
bad() { echo "preflight FAIL: $*" >&2; fail=1; }
[ "${driver%%.*}" -ge 580 ] || bad "driver $driver is below 580 (R-33)"
exp=${G1_EXPECT_GPU:-4090}; case "$gpu" in *"$exp"*) ;; *) bad "GPU is '$gpu', expected '$exp'" ;; esac
[ "$free_gb" -ge "$MIN_FREE" ] || bad "$WS has ${free_gb} GB free, need $MIN_FREE"
[ "${NVIDIA_DRIVER_CAPABILITIES:-}" = all ] || bad "NVIDIA_DRIVER_CAPABILITIES is '${NVIDIA_DRIVER_CAPABILITIES:-}', expected all"
[ -n "${POD_DEADLINE_EPOCH:-}" ] || bad "POD_DEADLINE_EPOCH is unset"

ok=true
[ "$fail" = 0 ] || ok=false
cat > "$G1_DIR/hostinfo.json" << JSON
{"gpu_name": "$gpu", "driver_version": "$driver", "cuda_version": "$cuda", "torch_cuda_version": "$torch_cuda", "vcpu": $vcpu, "ram_gb": $ram_gb, "workspace_free_gb": $free_gb, "env_ok": $ok}
JSON
echo "preflight: $gpu driver $driver cuda $cuda (torch $torch_cuda) ${vcpu} vCPU ${ram_gb} GB RAM"
exit "$fail"
