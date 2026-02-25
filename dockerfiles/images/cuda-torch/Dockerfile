# ── PyTorch layer ─────────────────────────────────────────────────
# Installs PyTorch + torchvision + torchaudio on top of the
# CUDA+TensorRT base image. This is the foundation for all
# application images.
#
# Build stable (2.10):
#   docker build --build-arg TORCH_VERSION=2.10.0 \
#       -t cuda13.0-torch2.10-ubuntu24.04 dockerfiles/images/cuda-torch/
#
# Build RC (2.11):
#   docker build --build-arg TORCH_VERSION=2.11.0 \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/test/cu130 \
#       -t cuda13.0-torch2.11rc1-ubuntu24.04 dockerfiles/images/cuda-torch/
#
# Build from source (slow, guaranteed ABI match):
#   docker build --build-arg TORCH_VERSION=2.11.0 \
#       --build-arg TORCHVISION_VERSION=0.26.0 \
#       --build-arg TORCHAUDIO_VERSION=2.11.0 \
#       --build-arg FORCE_BUILD=on \
#       -t cuda13.0-torch2.11-ubuntu24.04 dockerfiles/images/cuda-torch/
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
# Only needed for FORCE_BUILD=on (pip resolves automatically)
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0

# ── Install or build PyTorch + torchvision + torchaudio ──────────
#
# FORCE_BUILD=off (default):
#   Install official wheels from the PyTorch index.
#   pip resolves matching torchvision/torchaudio automatically.
#
# FORCE_BUILD=on:
#   Clone and build all three from source. Requires explicit
#   TORCHVISION_VERSION and TORCHAUDIO_VERSION args.
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
    \
    && echo "=== Building torchvision ${TORCHVISION_VERSION} from source ===" \
    && git clone --branch "v${TORCHVISION_VERSION}" --depth=1 \
    https://github.com/pytorch/vision /opt/vision \
    && cd /opt/vision \
    && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" FORCE_CUDA=1 \
    python3 setup.py bdist_wheel --dist-dir /opt/wheels \
    && uv pip install /opt/wheels/torchvision*.whl \
    && cd / && rm -rf /opt/vision /opt/wheels \
    \
    && echo "=== Building torchaudio ${TORCHAUDIO_VERSION} from source ===" \
    && git clone --branch "v${TORCHAUDIO_VERSION}" --depth=1 --recursive \
    https://github.com/pytorch/audio /opt/audio \
    && cd /opt/audio \
    && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" USE_CUDA=1 \
    python3 setup.py bdist_wheel --dist-dir /opt/wheels \
    && uv pip install /opt/wheels/torchaudio*.whl \
    && cd / && rm -rf /opt/audio /opt/wheels \
    ; else \
    echo "=== Installing PyTorch ${TORCH_VERSION} + ecosystem from pip ===" \
    && uv pip install \
    torch==${TORCH_VERSION}+${TORCH_CUDA} \
    torchvision torchaudio \
    --index-url ${TORCH_INDEX} \
    ; fi

# ── Environment ───────────────────────────────────────────────────
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TORCH_INDEX=${TORCH_INDEX}

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__} CUDA {torch.version.cuda}')" \
    && python -c "import torchvision; print(f'torchvision {torchvision.__version__}')" \
    && python -c "import torchaudio; print(f'torchaudio {torchaudio.__version__}')"

WORKDIR /opt
