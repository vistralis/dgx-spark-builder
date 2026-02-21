# ComfyUI on DGX Spark (Blackwell, SM 121)
#
# Built on the ABI-matched PyTorch base image for clean wheel compatibility.
#
# Usage:
#   # 1. Build the base image first
#   docker build -t cuda13.0-torch2.10-ubuntu24.04 dockerfiles/base/
#
#   # 2. Build wheels for base
#   BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 ./build_wheels.sh --staging sageattention bitsandbytes flash-attention
#   ./build_wheels.sh --promote pip-torch2.10-cu130
#
#   # 3. Build ComfyUI
#   docker build -f dockerfiles/comfyui/Dockerfile \
#       --build-context wheels=wheels/pip-torch2.10-cu130 .
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

# ── Pre-built ABI-matched wheels (explicit, by name) ─────────────
COPY wheels/sageattention-*.whl /tmp/wheels/
COPY wheels/bitsandbytes-*.whl /tmp/wheels/
COPY wheels/flash_attn-*.whl /tmp/wheels/
RUN uv pip install /tmp/wheels/*.whl && rm -rf /tmp/wheels

# ── Runtime config ────────────────────────────────────────────────
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
