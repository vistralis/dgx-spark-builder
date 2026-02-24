# ONNX Runtime

| | |
|---|---|
| **Repo** | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| **Version** | 1.25.0 |
| **Status** | ✅ Published |
| **Wheel** | `onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl` (55 MB) |

## What it does

Cross-platform inference engine for ONNX models. GPU variant includes CUDA Execution Provider and TensorRT Execution Provider for accelerated inference.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ |
| SM 121 | ✅ |
| aarch64 | ✅ (no prebuilt GPU wheels on PyPI for aarch64) |
| torch 2.10 | ✅ |
| torch 2.11 | ✅ |

## Build

```bash
docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    dockerfiles/builders/onnxruntime/
```

Build time: ~45 minutes. Requires TensorRT headers (present in cuda-tensorrt base image).
