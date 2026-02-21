# Build torchao wheel from source for Blackwell (SM 121)
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE} AS builder

ARG TORCHAO_VERSION=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV MAX_JOBS=20

WORKDIR /opt
RUN git clone --recursive https://github.com/pytorch/ao.git torchao \
    && cd torchao && git checkout ${TORCHAO_VERSION} \
    && git submodule update --init --recursive

WORKDIR /opt/torchao
RUN pip wheel . --no-build-isolation --no-deps --wheel-dir /wheels

FROM scratch
COPY --from=builder /wheels/ /
