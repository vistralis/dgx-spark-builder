# Build sageattention wheel from source for Blackwell (SM 121)
#
# SageAttention provides INT8/FP8 quantized attention kernels that can
# deliver ~2x speedup over SDPA for diffusion transformers.
#
# Usage (with ABI-matched base — recommended):
#   docker build -t cuda13.0-torch2.10-ubuntu24.04 dockerfiles/base/
#   docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
#                -t sage-builder dockerfiles/sageattention/
#
# Usage (with NGC base — may cause ABI mismatch):
#   docker build -t sage-builder dockerfiles/sageattention/
#
# Extract wheel:
#   docker run --rm -v $(pwd)/wheels/staging:/out sage-builder sh -c 'cp /*.whl /out/'
#
ARG BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04
FROM ${BASE_IMAGE} AS builder

ARG SAGE_VERSION=blackwell-sm121

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV MAX_JOBS=20

WORKDIR /opt
RUN git clone https://github.com/vistralis/SageAttention.git \
    && cd SageAttention && git checkout ${SAGE_VERSION}

WORKDIR /opt/SageAttention
RUN pip wheel . --no-build-isolation --no-deps --wheel-dir /wheels

FROM scratch
COPY --from=builder /wheels/ /
