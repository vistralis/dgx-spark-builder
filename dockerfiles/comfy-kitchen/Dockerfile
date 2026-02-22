# Build comfy-kitchen wheel with CUDA kernels for Blackwell (SM 120/121)
#
# comfy-kitchen is ComfyUI's fast kernel library. The PyPI wheel is
# pure-python (eager backend only). Building from source enables
# CUDA and Triton backends for significantly better performance.
#
# Usage:
#   docker build --no-cache -t comfy-kitchen-builder dockerfiles/comfy-kitchen/
#
ARG BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04
FROM ${BASE_IMAGE} AS builder

ARG COMFY_KITCHEN_VERSION=v0.2.7

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Build env ─────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"
ENV MAX_JOBS=20

# ── Clone and build ──────────────────────────────────────────────
WORKDIR /opt
RUN git clone https://github.com/Comfy-Org/comfy-kitchen.git \
    && cd comfy-kitchen && git checkout ${COMFY_KITCHEN_VERSION}

WORKDIR /opt/comfy-kitchen
RUN pip install nanobind scikit-build-core && \
    python setup.py build_ext --cuda-archs="120;121" bdist_wheel --dist-dir /wheels

# ── Output stage ──────────────────────────────────────────────────
FROM scratch
COPY --from=builder /wheels/ /
