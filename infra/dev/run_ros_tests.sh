#!/usr/bin/env bash
# Build and test ros2_ws in the Mac/CI dev container. Extra args go to colcon build and test.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ "${ROS_DEV_IMAGE_PREBUILT:-0}" != "1" ]; then
  docker build -t carla-sim-ros-dev -f infra/dev/Dockerfile.ros-dev .
fi
docker run --rm -v "$PWD":/ws -w /ws carla-sim-ros-dev bash -c '
set -eo pipefail  # no -u: ROS setup.bash reads unset variables
source /opt/ros/humble/setup.bash
uv pip install -e . --no-deps
cd ros2_ws
python -m colcon build --symlink-install "$@"
source install/setup.bash
python -m colcon test --event-handlers console_direct+ "$@"
python -m colcon test-result --verbose
' bash "$@"
