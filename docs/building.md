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
| `vllm` | ❌ No wheels | No aarch64 build, need SM 120/121 support |
| `comfy-kitchen` | ❌ No CUDA aarch64 | PyPI has eager-only; need CUDA+Triton backends |

## Prerequisites

- Docker with NVIDIA GPU support
- Custom base image built (e.g. `cuda13.0-torch2.10-ubuntu24.04`)
- ~20 GB disk for build containers

## Build Commands

```bash
# Build all wheels (default base: cuda13.0-torch2.10-ubuntu24.04)
./build_wheels.sh

# Build specific wheels
./build_wheels.sh sageattention flash-attention

# Build for PyTorch 2.11 RC
BASE_IMAGE=cuda13.0-torch2.11rc1-ubuntu24.04 ./build_wheels.sh vllm
```

## Output Layout

Output directory is auto-derived from the base image name:

```
wheels/
├── torch2.10-cu130/            # PyTorch 2.10 + CUDA 13.0
└── torch2.11rc1-cu130/         # PyTorch 2.11 RC1 + CUDA 13.0
```

See [WHEELS.md](../WHEELS.md) for the full inventory with source links and build commands.

## ABI Compatibility

> [!WARNING]
> Wheels are **NOT interchangeable** between base images. NGC PyTorch uses a custom
> C++ ABI that differs from pip-installed PyTorch. Always use wheels matching your target.

---

*Last updated: 2026-02-22*
