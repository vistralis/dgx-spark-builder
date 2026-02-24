# Build SGLang sgl-kernel + sglang wheels from source for Blackwell (SM 120/121)
#
# SGLang has two packages:
#   - sgl-kernel: CUDA kernels (scikit-build-core + CMake), produces sgl_kernel wheel
#   - python/:    Pure Python sglang package (setuptools-scm), produces sglang wheel
#
# The sgl-kernel CMakeLists.txt already has full SM 121a support for CUDA >= 13.0
# on aarch64 — no patches needed.
#
# Usage:
#   docker build --no-cache \
#     --build-arg BASE_IMAGE=cuda13.0-torch2.11rc1-ubuntu24.04 \
#     --build-arg SGLANG_VERSION=v0.5.9 \
#     -t sglang-builder:v0.5.9 \
#     dockerfiles/sglang/
#
# Extract wheels:
#   cid=$(docker create sglang-builder:v0.5.9); docker cp "$cid":/ /tmp/sgl/; docker rm "$cid"
#
ARG BASE_IMAGE=cuda13.0-torch2.11rc1-ubuntu24.04
FROM ${BASE_IMAGE} AS builder

ARG SGLANG_VERSION=v0.5.9
ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git curl ccache \
    gcc g++ \
    libnuma-dev libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Build env ─────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV MAX_JOBS=2
ENV CMAKE_BUILD_PARALLEL_LEVEL=2
ENV SGL_KERNEL_COMPILE_THREADS=2
# dlpack (mscclpp dep) uses cmake_minimum_required(VERSION 3.0)
# which cmake 4.2 rejects; this allows it
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# ── Clone SGLang ──────────────────────────────────────────────────
WORKDIR /opt
RUN git clone --depth=1 --branch ${SGLANG_VERSION} ${SGLANG_REPO} sglang

# ── Build sgl-kernel wheel ───────────────────────────────────────
# sgl-kernel uses scikit-build-core + CMake. It fetches CUTLASS,
# DeepGEMM, FlashInfer, flash-attention, mscclpp, triton via
# FetchContent at configure time.
WORKDIR /opt/sglang/sgl-kernel
RUN uv pip install scikit-build-core setuptools wheel numpy && \
    pip wheel . --no-deps --wheel-dir /wheels

# ── Build sglang Python wheel ────────────────────────────────────
# Pure Python package with setuptools-scm for versioning.
WORKDIR /opt/sglang/python
RUN uv pip install setuptools-scm && \
    pip wheel . --no-deps --wheel-dir /wheels

# ── Output stage (scratch = wheel files only) ─────────────────────
FROM scratch
COPY --from=builder /wheels/ /
