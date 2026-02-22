# ── PyTorch layer ─────────────────────────────────────────────────
# Installs PyTorch on top of the CUDA+TensorRT base image.
# This is the "light" layer (~2 GB) that changes when swapping
# PyTorch versions.
#
# Build stable (2.10):
#   docker build --build-arg TORCH_VERSION=2.10.0 \
#       -t cuda13.0-torch2.10-ubuntu24.04 dockerfiles/base-cuda-torch-ubuntu/
#
# Build RC (2.11):
#   docker build --build-arg TORCH_VERSION=2.11.0 \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/test/cu130 \
#       -t cuda13.0-torch2.11rc1-ubuntu24.04 dockerfiles/base-cuda-torch-ubuntu/
#
# Build from source (slow, guaranteed ABI match):
#   docker build --build-arg TORCH_VERSION=2.11.0 \
#       --build-arg FORCE_BUILD=on \
#       -t cuda13.0-torch2.11-ubuntu24.04 dockerfiles/base-cuda-torch-ubuntu/
#
# ─────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=cuda13.0-tensorrt-ubuntu24.04
FROM ${BASE_IMAGE}

# ── Build args ────────────────────────────────────────────────────
ARG TORCH_VERSION=2.10.0
ARG TORCH_CUDA=cu130
ARG TORCH_INDEX=https://download.pytorch.org/whl/${TORCH_CUDA}
ARG TORCH_CUDA_ARCH_LIST="12.0;12.1"
ARG FORCE_BUILD=off

# ── Install or build PyTorch ──────────────────────────────────────
#
# FORCE_BUILD=off (default):
#   Install the official PyTorch wheel from the PyTorch index.
#   Fast (~30s).
#
# FORCE_BUILD=on:
#   Clone and build PyTorch from source. Slow (~2-4 hours) but
#   guarantees the exact same build configuration.
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
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__} CUDA {torch.version.cuda}')"

WORKDIR /opt
