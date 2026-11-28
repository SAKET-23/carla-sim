#!/usr/bin/env bash
# Pod entrypoint (image-and-bootstrap.md section 3). Runs as root on every start; idempotent.
# Steps 1-6 log and continue on failure: it must always reach `exec sleep infinity`, or the pod restart-loops
# with no SSH.
set -euo pipefail

mkdir -p /workspace/logs 2>/dev/null || true
if [ -w /workspace/logs ]; then
  exec > >(tee -a "/workspace/logs/entrypoint-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
fi
warn() { echo "entrypoint: $*" >&2; }

# 1. sshd. ECC-29: PUBLIC_KEY is set on the template (custom images do not get it injected).
step_ssh() {
  ssh-keygen -A
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  local key="${PUBLIC_KEY:-}"
  if [ -n "$key" ]; then
    grep -qxF "$key" /root/.ssh/authorized_keys || echo "$key" >> /root/.ssh/authorized_keys
  else
    warn "PUBLIC_KEY is not set: no SSH login possible"
  fi
  mkdir -p /run/sshd
  /usr/sbin/sshd
}

# 2. /workspace layout (runpod-setup.md section 2)
step_layout() {
  mkdir -p /workspace/{carla/downloads,datasets/det,datasets/seg,models/onnx,models/engines,models/checkpoints,runs,bags,venvs,logs}
  chown -R carla:carla /workspace/carla /workspace/logs
}

# 3. device-node group for the carla user
step_render_group() {
  local node group
  node=$(compgen -G '/dev/dri/renderD*' | head -n 1 || true)
  if [ -n "$node" ]; then
    group=$(stat -c %G "$node")
    if [ "$group" != "UNKNOWN" ]; then usermod -aG "$group" carla; fi
  fi
}

# 4. repo clone on the volume
step_repo() {
  if [ -d /workspace/carla-sim ]; then
    ln -sfn /workspace/runs /workspace/carla-sim/runs
    uv pip install -e /workspace/carla-sim --no-deps
  fi
}

# 5. environment for SSH sessions (they do not inherit the image ENV or the RunPod variables). ECC-29
step_profile() {
  cat > /etc/profile.d/carla-sim.sh <<'PROFILE'
export NVIDIA_DRIVER_CAPABILITIES=all
export CARLA_ROOT=/workspace/carla/0.9.16
export PYTHONPATH=/workspace/carla/0.9.16/PythonAPI/carla
export VIRTUAL_ENV=/opt/venv
export PATH=/opt/venv/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/opt/venv
export UV_NO_SYNC=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LANG=C.UTF-8 LC_ALL=C.UTF-8
if [ -r /etc/carla-sim.env ]; then . /etc/carla-sim.env; fi
if [ -f /opt/ros/humble/setup.bash ] && [ -n "${BASH_VERSION:-}" ]; then
  . /opt/ros/humble/setup.bash
  if [ -f /opt/kiss_ws/install/setup.bash ]; then . /opt/kiss_ws/install/setup.bash; fi
  if [ -f /workspace/carla-sim/ros2_ws/install/setup.bash ]; then . /workspace/carla-sim/ros2_ws/install/setup.bash; fi
fi
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
PROFILE
  chmod 644 /etc/profile.d/carla-sim.sh
  # RunPod runtime variables and secrets, written at boot from this container's own environment (never baked)
  local old_umask var
  old_umask=$(umask)
  umask 077
  : > /etc/carla-sim.env
  for var in $(compgen -e | grep -E '^(RUNPOD_|G1_|SMOKE_|POD_DEADLINE_EPOCH$|ANTHROPIC_API_KEY$|CARLA_QUALITY$)' || true); do
    printf 'export %s=%q\n' "$var" "${!var}" >> /etc/carla-sim.env
  done
  # numpy's BLAS sizes its pool to the host's cores (128), not the pod's CPU quota: ~250 spinning threads per process
  echo 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1' >> /etc/carla-sim.env
  umask "$old_umask"
  chmod 600 /etc/carla-sim.env
}

# 6. status block
step_status() {
  echo "=== carla-sim pod ==="
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
  echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"
  if [ -f "${CARLA_ROOT:-/workspace/carla/0.9.16}/.installed" ]; then echo "CARLA: installed"; else echo "CARLA: not installed (run install_carla.sh)"; fi
  if [ -f /workspace/carla-sim/ros2_ws/install/setup.bash ]; then echo "ros2_ws: built"; else echo "ros2_ws: not built"; fi
  echo "runpodctl: $(command -v runpodctl || echo missing), RUNPOD_POD_ID=${RUNPOD_POD_ID:-unset}"
  df -h /workspace | tail -n 1
}

for s in step_ssh step_layout step_render_group step_repo step_profile step_status; do
  "$s" || warn "$s failed (continuing)"
done

# 7. keep the pod alive; CARLA is never auto-started
exec sleep infinity
