# Build FlashInfer wheel for Blackwell (SM 120/121)
#
# FlashInfer is a high-performance attention kernel library used as
# the default attention backend in vLLM. No prebuilt aarch64 wheels
# exist, so we build from source.
#
# Usage:
#   docker build --no-cache -t flashinfer-builder dockerfiles/flashinfer/
#
ARG BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04
FROM ${BASE_IMAGE} AS builder

ARG FLASHINFER_VERSION=v0.6.4

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git curl \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Build env ─────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"
ENV MAX_JOBS=16
ENV FLASHINFER_ENABLE_SM120=1

# ── Clone with submodules ────────────────────────────────────────
WORKDIR /opt
RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git \
    && cd flashinfer && git checkout ${FLASHINFER_VERSION} \
    && git submodule update --init --recursive

# ── Install build deps and build wheel ───────────────────────────
WORKDIR /opt/flashinfer
RUN pip install ninja numpy && \
    pip wheel . --no-deps --no-build-isolation --wheel-dir /wheels

# ── Output stage ──────────────────────────────────────────────────
FROM scratch
COPY --from=builder /wheels/ /
