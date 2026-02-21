# Build onnxruntime-gpu wheel from source for Blackwell (SM 121)
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${BASE_IMAGE} AS builder

ARG ORT_REPO=microsoft/onnxruntime
ARG ORT_VERSION=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.1"

WORKDIR /opt
RUN git clone --recursive https://github.com/${ORT_REPO}.git onnxruntime \
    && cd onnxruntime && git checkout ${ORT_VERSION} \
    && git submodule update --init --recursive

WORKDIR /opt/onnxruntime
RUN ./build.sh \
    --config Release \
    --build_dir build/cuda \
    --parallel 20 \
    --nvcc_threads 20 \
    --allow_running_as_root \
    --use_cuda \
    --cuda_version $(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+') \
    --cuda_home ${CUDA_HOME} \
    --cudnn_home ${CUDA_HOME} \
    --use_tensorrt \
    --tensorrt_home /usr \
    --build_wheel \
    --skip_tests \
    --cmake_generator Ninja \
    --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=121 \
    onnxruntime_BUILD_UNIT_TESTS=OFF

RUN mkdir -p /wheels && \
    cp build/cuda/Release/dist/onnxruntime*.whl /wheels/

FROM scratch
COPY --from=builder /wheels/ /
