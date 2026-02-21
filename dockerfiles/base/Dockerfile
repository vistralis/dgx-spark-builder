# ── ABI-matched builder base ──────────────────────────────────────
# Provides a clean Ubuntu + pip-installed PyTorch build environment
# so CUDA extensions (SageAttention, FlashAttention, xformers, etc.)
# are compiled against the *exact same* PyTorch binary used at
# inference time.
#
# The NGC pytorch image has its own custom-built PyTorch that may
# differ at the C++ ABI level from pip-installed PyTorch, causing
# "Cannot access data pointer of Tensor that doesn't have storage"
# errors when loading Docker-built .so files in a pip-based venv.
#
# This image uses:
#   - nvidia/cuda cudnn-devel image (CUDA 13.0, Ubuntu 24.04, aarch64)
#   - TensorRT 10.14 + cuSPARSELt + upgraded cuDNN/cuBLAS/NCCL
#   - PyTorch built from source (or installed from pip wheel)
#
# This is intended as a drop-in replacement for NGC pytorch images,
# providing a clean pip-based build environment with the full NVIDIA
# inference stack.
#
# Usage (pip wheel — fast, default):
#   docker build -t cuda13.0-torch2.10-ubuntu24.04 dockerfiles/base/
#
# Usage (from source — slow but guaranteed ABI match):
#   docker build --build-arg FORCE_BUILD=on \
#                -t cuda13.0-torch2.10-ubuntu24.04 dockerfiles/base/
#
# Then reference in extension Dockerfiles:
#   FROM cuda13.0-torch2.10-ubuntu24.04 AS builder
# ─────────────────────────────────────────────────────────────────
ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

# ── Build args ────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.10.0
ARG TORCH_CUDA=cu130
ARG TORCH_INDEX=https://download.pytorch.org/whl/${TORCH_CUDA}
ARG TORCH_CUDA_ARCH_LIST="12.0;12.1"
ARG FORCE_BUILD=off

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
# The NVIDIA repo also has cuda13.1 builds; pin to avoid conflicts.
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

# ── Install TensorRT Python bindings ─────────────────────────────
# apt tensorrt-dev provides C headers/libs; pip provides Python API.
# Only install bindings — the libs are already present from apt.
RUN uv pip install tensorrt-cu13-bindings==10.14.1.48

# ── Install or build PyTorch ──────────────────────────────────────
#
# FORCE_BUILD=off (default):
#   Install the official PyTorch wheel from the PyTorch index.
#   Fast (~30s) but the wheel may have a different ABI than your
#   runtime if you built PyTorch from source on the host.
#
# FORCE_BUILD=on:
#   Clone and build PyTorch from source. Slow (~2-4 hours) but
#   guarantees the exact same build configuration.
#   Ref: https://github.com/dusty-nv/jetson-containers
# ──────────────────────────────────────────────────────────────────
RUN if [ "${FORCE_BUILD}" = "on" ]; then \
    echo "=== Building PyTorch ${TORCH_VERSION} from source ===" \
    && git clone --branch "v${TORCH_VERSION}" --depth=1 --recursive \
    https://github.com/pytorch/pytorch /opt/pytorch \
    && cd /opt/pytorch \
    && uv pip install -r requirements.txt \
    && export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    && export TORCH_NVCC_FLAGS="-Xfatbin -compress-all -compress-mode=size" \
    && export USE_PRIORITIZED_TEXT_FOR_LD=1 \
    && export PYTORCH_BUILD_VERSION=${TORCH_VERSION} \
    && export PYTORCH_BUILD_NUMBER=1 \
    && export USE_CUDA=1 \
    && export USE_CUDNN=1 \
    && export USE_CUSPARSELT=1 \
    && export USE_NCCL=1 \
    && export USE_DISTRIBUTED=1 \
    && export USE_FLASH_ATTENTION=1 \
    && export USE_MEM_EFF_ATTENTION=1 \
    && export USE_NATIVE_ARCH=0 \
    && export USE_FBGEMM=0 \
    && export USE_TENSORRT=0 \
    && python3 setup.py bdist_wheel --dist-dir /opt/wheels \
    && uv pip install /opt/wheels/torch*.whl \
    && cd / && rm -rf /opt/pytorch /opt/wheels \
    ; else \
    echo "=== Installing PyTorch ${TORCH_VERSION} from pip ===" \
    && uv pip install \
    torch==${TORCH_VERSION}+${TORCH_CUDA} \
    --index-url ${TORCH_INDEX} \
    ; fi

# ── Environment ───────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV MAX_JOBS=20
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__} CUDA {torch.version.cuda}'); \
    import tensorrt; print(f'TensorRT {tensorrt.__version__}'); \
    import numpy; print(f'NumPy {numpy.__version__}')"

WORKDIR /opt
