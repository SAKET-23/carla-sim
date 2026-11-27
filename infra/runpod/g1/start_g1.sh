#!/usr/bin/env bash
# Start the G1 tmux session (deadline watchdog, heartbeat, runner). Idempotent. Windows use a login shell
# so they see the pod variables from /etc/profile.d.
set -euo pipefail

D=$(cd "$(dirname "$0")" && pwd)
tmux has-session -t g1 && { echo "start_g1: session g1 already exists"; exit 0; }
tmux new-session -d -s g1 -n deadline "bash -lc 'bash $D/deadline.sh'"
tmux new-window -t g1 -n heartbeat "bash -lc 'bash $D/heartbeat.sh'"
tmux new-window -t g1 -n runner "bash -lc 'bash $D/run_g1.sh; exec bash'"
echo "start_g1: started session g1"
