#!/usr/bin/env bash
# Build the OpenPCDet training venv on the network volume (image-and-bootstrap.md section 5). Phase G3 only.
# Built on the pod because its CUDA ops must be compiled for the actual GPU architecture.
set -euo pipefail

mkdir -p /workspace/logs
exec > >(tee -a "/workspace/logs/install_openpcdet-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
die() { echo "install_openpcdet: $*" >&2; exit 1; }

VENV=/workspace/venvs/pcdet
SRC=/workspace/venvs/src
# Pinned commits: newest master at implementation time (2026-11-19)
PCDET_COMMIT=233f849829b6ac19afb8af8837a0246890908755
POINTPILLARS_COMMIT=ce7e2bd694c90207435c8751d61cdb38d48a9f4c
# ECC-28: spconv wheel name must match the CUDA minor of the chosen torch build; override with SPCONV_PKG
SPCONV_PKG="${SPCONV_PKG:-spconv-cu126}"

command -v nvcc > /dev/null || die "nvcc not found (the image's CUDA devel base provides it)"
mkdir -p "$SRC"

# 1-2. venv, torch (same pin and index as the runtime), spconv
uv venv "$VENV" --python /usr/bin/python3.10
export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT
uv pip install "torch==2.14.1" "numpy<2" "$SPCONV_PKG"

# 3. OpenPCDet at the pinned commit; setup.py develop was removed in setuptools 80
if [ ! -d "$SRC/OpenPCDet" ]; then
  git clone https://github.com/open-mmlab/OpenPCDet "$SRC/OpenPCDet"
fi
git -C "$SRC/OpenPCDet" checkout "$PCDET_COMMIT"
uv pip install "setuptools<80" wheel
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
[ -n "$CAP" ] || die "cannot read the GPU compute capability (nvidia-smi)"
uv pip install -r "$SRC/OpenPCDet/requirements.txt"
(cd "$SRC/OpenPCDet" && TORCH_CUDA_ARCH_LIST="$CAP" python setup.py develop)

# 4. smoke test
python -c "import pcdet; from pcdet.ops.iou3d_nms import iou3d_nms_utils; print('pcdet ok')"

# 5. CUDA-PointPillars: the ONNX exporter script only (inference uses perception/detection_trt.py)
if [ ! -d "$SRC/CUDA-PointPillars" ]; then
  git clone https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars "$SRC/CUDA-PointPillars"
fi
git -C "$SRC/CUDA-PointPillars" checkout "$POINTPILLARS_COMMIT"
echo "OpenPCDet venv ready: $VENV"
