# ComfyUI on DGX Spark (Blackwell, SM 121)
#
# Built on the ABI-matched PyTorch base image for clean wheel compatibility.
# Pre-built wheels are fetched from the GitHub release.
#
# Usage:
#   docker build -f dockerfiles/comfyui/Dockerfile -t comfyui-spark .
#
ARG BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04
FROM ${BASE_IMAGE}

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 libsndfile1 sox \
    libglfw3-dev libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# ── torchaudio + torchvision from PyTorch cu130 index ─────────────
RUN uv pip install torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# ── ComfyUI from source ──────────────────────────────────────────
ARG COMFYUI_VERSION=master
RUN git clone --depth=1 --branch ${COMFYUI_VERSION} \
    https://github.com/comfyanonymous/ComfyUI.git /opt/comfyui

WORKDIR /opt/comfyui
RUN uv pip install -r requirements.txt

# ── Pre-built ABI-matched wheels from GitHub release ─────────────
ARG WHEEL_BASE=https://github.com/vistralis/dgx-spark-builder/releases/download/pip-torch2.10-cu130-v0.1.0
RUN uv pip install \
    ${WHEEL_BASE}/sageattention-2.2.0-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/bitsandbytes-0.50.0.dev0-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl

# ── Runtime config ────────────────────────────────────────────────
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
