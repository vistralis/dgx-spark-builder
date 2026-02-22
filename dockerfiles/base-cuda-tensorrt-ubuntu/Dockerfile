# ── CUDA + TensorRT base image ────────────────────────────────────
# Provides a clean Ubuntu + NVIDIA inference stack (TensorRT, cuDNN,
# cuBLAS, NCCL, cuSPARSELt) with Python, uv, and build tools.
#
# This is the "heavy" layer (~10 GB) that rarely changes. PyTorch
# is installed separately via dockerfiles/pytorch/Dockerfile so that
# swapping PyTorch versions takes seconds instead of re-downloading
# 5 GB of TensorRT.
#
# Build:
#   docker build -t cuda13.0-tensorrt-ubuntu24.04 dockerfiles/base-cuda-tensorrt-ubuntu/
#
# Then layer PyTorch on top:
#   docker build --build-arg TORCH_VERSION=2.10.0 \
#       -t cuda13.0-torch2.10-ubuntu24.04 dockerfiles/base-cuda-torch-ubuntu/
#
# ─────────────────────────────────────────────────────────────────
ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

# ── Build args ────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.12

# ── System deps ───────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    cmake ninja-build git curl ca-certificates \
    libomp-dev \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── NVIDIA inference stack ────────────────────────────────────────
# TensorRT + cuSPARSELt + upgrade cuDNN/cuBLAS/NCCL/nvjitlink
# Pin TensorRT to cuda13.0 version matching our CUDA toolkit.
ARG TENSORRT_VERSION=10.14.1.48-1+cuda13.0
RUN printf 'Package: libnvinfer* libnvonnxparsers* tensorrt*\nPin: version *+cuda13.0\nPin-Priority: 1001\n' \
    > /etc/apt/preferences.d/tensorrt-cuda13 \
    && apt-get update && apt-get install -y --no-install-recommends \
    tensorrt-dev=${TENSORRT_VERSION} \
    libcusparselt0-dev-cuda-13 \
    && apt-get upgrade -y --allow-change-held-packages --no-install-recommends \
    libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    libcublas-13-0 libcublas-dev-13-0 \
    libnccl2 libnccl-dev \
    libnvjitlink-13-0 libnvjitlink-dev-13-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Install uv ────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

# ── Create build venv ─────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv ${VIRTUAL_ENV} --python python${PYTHON_VERSION}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# ── Install build tools ──────────────────────────────────────────
RUN uv pip install pip setuptools wheel scikit-build ninja 'cmake<4' numpy

# ── Environment ───────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV MAX_JOBS=20

# Verify
RUN python -c "import numpy; print(f'NumPy {numpy.__version__}')" \
    && ls /usr/lib/aarch64-linux-gnu/libnvinfer.so && echo 'TensorRT libs OK'

WORKDIR /opt
