# Build bitsandbytes wheel from source for Blackwell (SM 121)
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE} AS builder

ARG BNB_VERSION=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV BNB_CUDA_VERSION=130

WORKDIR /opt
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git \
    && cd bitsandbytes && git checkout ${BNB_VERSION}

WORKDIR /opt/bitsandbytes
RUN pip install --no-cache-dir scikit-build-core setuptools cmake ninja
RUN pip wheel . --no-build-isolation --no-deps --wheel-dir /wheels

FROM scratch
COPY --from=builder /wheels/ /
