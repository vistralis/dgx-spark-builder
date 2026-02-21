# Build flash-attn wheel from source for Blackwell (SM 121)
#
# ⚠ NOTE: Flash Attention 2 on GB10 (SM 121) is ~2% SLOWER than PyTorch SDPA.
# The kernels compile and run but aren't optimized for Blackwell's architecture.
# See: https://github.com/Dao-AILab/flash-attention/issues/1969
#
# This build is provided for completeness and future compatibility.
# FA3/FA4 native Blackwell support is not yet available.
#
# Usage:
#   docker build -t flash-builder dockerfiles/flash-attention/
#   docker run --rm -v $(pwd)/wheels/staging:/out flash-builder sh -c 'cp /*.whl /out/'
#
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE} AS builder

ARG FLASH_ATTN_VERSION=v2.8.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV MAX_JOBS=4
ENV NVCC_THREADS=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=4
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

WORKDIR /opt
RUN git clone --depth=1 --branch ${FLASH_ATTN_VERSION} \
    https://github.com/Dao-AILab/flash-attention.git

WORKDIR /opt/flash-attention
RUN pip wheel . -v --no-deps --no-build-isolation --wheel-dir /wheels

FROM scratch
COPY --from=builder /wheels/ /
