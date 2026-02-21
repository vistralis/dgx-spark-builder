# Build vLLM wheel from source for Blackwell (SM 120/121)
#
# vLLM is a high-throughput LLM serving engine. Building from source
# ensures ABI compatibility with our PyTorch build and enables
# Blackwell-specific CUDA kernels.
#
# Usage:
#   docker build --no-cache -t vllm-builder dockerfiles/vllm/
#
# Extract wheel:
#   docker run --rm -v $(pwd)/wheels/staging:/out vllm-builder sh -c 'cp /*.whl /out/'
#
ARG BASE_IMAGE=ubuntu2404-pt210-cu130
FROM ${BASE_IMAGE} AS builder

ARG VLLM_VERSION=v0.16.0

# ── System deps for vLLM build ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git curl \
    gcc g++ \
    libnuma-dev libucx-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Build env ─────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"
ENV MAX_JOBS=16
ENV VLLM_TARGET_DEVICE=cuda
ENV VLLM_VERSION_OVERRIDE="0.16.0+cu130"
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.16.0+cu130"
ENV VLLM_DOCKER_BUILD_CONTEXT=1
ENV VLLM_SKIP_CUPY_CHECK=1
ENV TRITON_OFFLINE=1
ENV CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"

# ── Clone vLLM ────────────────────────────────────────────────────
WORKDIR /opt
RUN git clone --depth=1 --branch ${VLLM_VERSION} \
    https://github.com/vllm-project/vllm.git

# ── Install build + runtime dependencies ──────────────────────────
# vLLM 0.16.0 pins torch==2.10.0 which matches our base image,
# so no torch filtering is needed (unlike v0.15.1 which pinned 2.9.1).
WORKDIR /opt/vllm
RUN uv pip install numpy setuptools-scm && \
    uv pip install -r requirements/build.txt && \
    if [ -f requirements/common.txt ]; then \
    uv pip install -r requirements/common.txt; \
    fi

# ── Build wheel ───────────────────────────────────────────────────
RUN python3 setup.py bdist_wheel --dist-dir /wheels

# ── Output stage (scratch = wheel files only) ─────────────────────
FROM scratch
COPY --from=builder /wheels/ /
