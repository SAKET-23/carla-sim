# syntax=docker/dockerfile:1
# RunPod pod image (Plan/04-gpu/image-and-bootstrap.md section 2). Build with --platform linux/amd64.
# CARLA, the repo and the OpenPCDet venv live on the network volume, not here.
# ECC-28: base tag = newest CUDA 12.x with -cudnn-devel-ubuntu22.04 (nvcc for OpenPCDet); the torch 2.14.1 wheel
# brings its own CUDA runtime, so the pins are only proven together by the pin-check RUN at the end.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CARLA_ROOT=/workspace/carla/0.9.16
ENV PYTHONPATH=/workspace/carla/0.9.16/PythonAPI/carla
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Base packages (iproute2: ss, procps: pgrep/pkill, both used by start_carla.sh / stop_carla.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget gnupg lsb-release software-properties-common git git-lfs rsync tmux htop nano \
    unzip pv openssh-server sudo locales iproute2 procps build-essential cmake pkg-config \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# CARLA runtime libraries. Never install the NVIDIA driver GL packages: the host mounts the driver libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsdl2-2.0-0 xserver-xorg libomp5 xdg-user-dirs libegl1 libgl1 libglu1-mesa libjpeg-turbo8 libtiff5 \
    libpng16-16 fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Vulkan. mesa-vulkan-drivers (lavapipe) is here only so smoke 02 can detect a software fallback by name.
# The ICD file is a fallback for when the container toolkit does not inject one.
RUN apt-get update && apt-get install -y --no-install-recommends libvulkan1 vulkan-tools mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/* \
    && if [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then \
         mkdir -p /etc/vulkan/icd.d \
         && printf '%s\n' '{' '    "file_format_version" : "1.0.0",' '    "ICD": {' \
              '        "library_path": "libGLX_nvidia.so.0",' '        "api_version" : "1.3.0"' '    }' '}' \
              > /etc/vulkan/icd.d/nvidia_icd.json; \
       fi

# ROS 2 Humble
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base ros-humble-rmw-fastrtps-cpp ros-humble-vision-msgs ros-humble-foxglove-bridge \
    ros-humble-tf2-ros ros-humble-tf2-tools ros-humble-rosbag2 ros-humble-rosbag2-storage-mcap \
    ros-humble-sensor-msgs-py ros-humble-launch-testing ros-humble-launch-testing-ament-cmake \
    ros-humble-topic-tools python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/* \
    && rosdep init

# KISS-ICP ROS 2 wrapper (not on apt). C++ only, so the bare colcon with system Python is fine before the venv exists.
# Its system dependencies come from rosdep.
WORKDIR /opt/kiss_ws
RUN git clone --depth 1 --branch v1.3.0 https://github.com/PRBonn/kiss-icp.git src/kiss-icp \
    && apt-get update && rosdep update \
    && rosdep install --from-paths src --ignore-src -y --rosdistro humble \
    && bash -c 'source /opt/ros/humble/setup.bash && colcon build --packages-select kiss_icp --cmake-args -DCMAKE_BUILD_TYPE=Release' \
    && rm -rf /var/lib/apt/lists/* /opt/kiss_ws/build /opt/kiss_ws/log /root/.ros

COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/

# Python venv: system site packages expose rclpy (via PYTHONPATH from setup.bash) and apt's colcon/setuptools 59.6.
# Dependencies only, from the lock file, so this layer is cached independently of the source.
RUN uv venv /opt/venv --python /usr/bin/python3.10 --system-site-packages
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra gpu --extra llm --extra dev
# Humble tooling constraints (same as Dockerfile.ros-dev): the venv must not carry its own setuptools (colcon
# --symlink-install needs apt's 59.6), and launch_testing's pytest plugin fails under pytest 9.
RUN uv pip uninstall setuptools && uv pip install "pytest<9"
RUN python -c "import numpy, setuptools; assert numpy.__version__.startswith('1.'), numpy.__version__; assert setuptools.__version__.startswith('59.'), setuptools.__version__"

# Users: UE4 refuses root, so CARLA runs as uid 1000; everything else (sshd, apt, RunPod start-up) stays root.
# sshd: key-only root login (ECC-29: PUBLIC_KEY is written by the entrypoint).
RUN useradd -m -u 1000 -s /bin/bash carla \
    && usermod -aG video carla \
    && mkdir -p /run/sshd /etc/ssh/sshd_config.d \
    && printf '%s\n' 'PermitRootLogin prohibit-password' 'PasswordAuthentication no' > /etc/ssh/sshd_config.d/10-pod.conf

COPY --chmod=755 infra/runpod/entrypoint.sh infra/runpod/install_carla.sh infra/runpod/start_carla.sh \
     infra/runpod/stop_carla.sh infra/runpod/install_openpcdet.sh infra/runpod/autostop.sh /usr/local/bin/
COPY --chmod=755 infra/runpod/smoke/ /usr/local/share/carla-sim/smoke/

# ECC-28: the build fails if the pins do not import together (no GPU needed for the imports)
RUN bash -c 'source /opt/ros/humble/setup.bash && python -c "import torch, tensorrt, onnxruntime, carla, rclpy, numpy; print(torch.__version__, torch.version.cuda, tensorrt.__version__, onnxruntime.__version__, numpy.__version__)"'

WORKDIR /
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
