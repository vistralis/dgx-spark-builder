# Custom Wheel Building for DGX Spark (aarch64)

Several critical packages have no prebuilt wheels for aarch64 Linux on PyPI.
We build them from source inside Docker containers.

## Why Custom Wheels?

| Package | PyPI Status (aarch64) | Why |
|---------|----------------------|-----|
| `onnxruntime-gpu` | ❌ No wheels | Only x86_64 and Windows wheels published |
| `bitsandbytes` | ❌ No CUDA aarch64 | CPU-only aarch64 wheels, need CUDA build |
| `sageattention` | ❌ No wheels | Our SM 121 fork, not on PyPI |
| `flash-attn` | ❌ No wheels | No aarch64 CUDA wheels on PyPI |

## Prerequisites

- Docker with NVIDIA GPU support
- NGC container access (`nvcr.io/nvidia/pytorch`) or custom base image built
- ~20 GB disk for build containers

## Build Commands

```bash
# Build all wheels (default targets: bitsandbytes, sageattention, flash-attention, onnxruntime)
./build_wheels.sh

# Build specific wheel
./build_wheels.sh onnxruntime

# Use custom base image (ABI-matched pip PyTorch)
BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 ./build_wheels.sh sageattention

# Use NGC base image
BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 ./build_wheels.sh

# Build to staging first
./build_wheels.sh --staging sageattention

# Promote tested wheels
./build_wheels.sh --promote pip-torch2.10-cu130
```

## Output Layout

```
wheels/
├── staging/                  # Temp build output
├── ngc-25.11-py3/            # Wheels built on NGC 25.11
├── ngc-26.01-py3/            # Wheels built on NGC 26.01
└── pip-torch2.10-cu130/      # Wheels built on custom base
```

## ABI Compatibility

> [!WARNING]
> Wheels are **NOT interchangeable** between base images. NGC PyTorch uses a custom
> C++ ABI that differs from pip-installed PyTorch. Always use wheels matching your target.

---

*Last updated: 2026-02-21*
