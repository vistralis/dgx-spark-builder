# DGX Spark Builder

Build infrastructure for **DGX Spark GB10** (Blackwell, SM 121, aarch64):
custom CUDA wheel builds, Docker images for ML workloads.

## Hardware Target

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell) |
| Compute Capability | SM 12.0 / 12.1 |
| CPU Architecture | aarch64 (Grace) |
| CUDA | 13.0 / 13.1 |
| OS | Ubuntu 24.04 |
| Memory | 128 GB unified (VMM: yes, can address ~160 GB) |
| Host Driver | CUDA 13.0 native |

## What This Repo Does

1. **Builds custom wheels** for packages that have no prebuilt aarch64+CUDA wheels on PyPI
2. **Provides Dockerfiles** for ML application images (ComfyUI, vLLM, ZImage pipeline)
3. **Organizes wheels by ABI target** so the right wheels are installed in the right images

## Why Custom Wheels?

Several critical ML packages ship no aarch64+CUDA wheels on PyPI:

| Package | PyPI Status (aarch64) | Our Solution |
|---------|----------------------|---------------|
| `onnxruntime-gpu` | ❌ No wheels | Build from source with CUDA+TensorRT |
| `bitsandbytes` | ❌ CPU-only | Build from source with CUDA 13.0 |
| `sageattention` | ❌ No wheels | Build from our SM 121 fork |
| `flash-attn` | ❌ No wheels | Build from source (SM 121) |
| `xformers` | ❌ No wheels | Build from source (SM 121, currently disabled) |
| `torchao` | ❌ No wheels | Build from source (SM 121, currently disabled) |
| `vllm` | ❌ No aarch64 | Build from source for SM 120/121 |

## Quick Start

```bash
# 1. Build the ABI-matched PyTorch base image
docker build -t ubuntu2404-pt210-cu130 dockerfiles/base/

# 2. Build wheels for the custom base
BASE_IMAGE=ubuntu2404-pt210-cu130 ./build_wheels.sh --staging sageattention bitsandbytes

# 3. Promote tested wheels
./build_wheels.sh --promote pip-pt210-cu130

# 4. Build an application image (e.g., ComfyUI)
docker build -f dockerfiles/comfyui/Dockerfile \
    --build-context wheels=wheels/pip-pt210-cu130 .
```

## Three Base Image Targets

| Target | Tag | Use Case |
|--------|-----|----------|
| **NGC PyTorch 25.11** | `nvcr.io/nvidia/pytorch:25.11-py3` | Matches host driver, batteries-included |
| **NGC PyTorch 26.01** | `nvcr.io/nvidia/pytorch:26.01-py3` | Latest NGC, forward-compat drivers |
| **Custom pip base** | `ubuntu2404-pt210-cu130` | Lean, ABI-clean pip-installed PyTorch |

> **ABI Warning**: Wheels built on NGC images are NOT compatible with the custom pip base
> (and vice versa). NGC uses a custom PyTorch build with different C++ ABI.
> Always use wheels matching your base image.

## Wheels Directory Layout

```
wheels/
├── staging/                  # Temp build output (not committed)
├── ngc-25.11-py3/            # Wheels built on NGC 25.11
│   ├── onnxruntime_gpu-*.whl
│   ├── bitsandbytes-*.whl
│   └── sageattention-*.whl
├── ngc-26.01-py3/            # Wheels built on NGC 26.01
│   └── onnxruntime_gpu-*.whl
└── pip-pt210-cu130/          # Wheels built on custom base
    ├── sageattention-*.whl
    ├── flash_attn-*.whl
    └── bitsandbytes-*.whl
```

Wheels are published as **GitHub Releases** for easy download without cloning.

## Project Structure

```
dgx-spark-builder/
├── README.md                 # This file
├── CONTEXT.md                # Seed context for AI agents working on this repo
├── artifacts.yaml            # Build matrix config
├── build_artifacts.py        # Python build orchestrator
├── build_wheels.sh           # Bash build script
├── probe_images.sh           # NGC image probing utility
├── dockerfiles/
│   ├── base/Dockerfile       # ABI-matched PyTorch base image
│   ├── comfyui/Dockerfile    # ComfyUI image
│   ├── onnxruntime/Dockerfile
│   ├── bitsandbytes/Dockerfile
│   ├── sageattention/Dockerfile
│   ├── flash-attention/Dockerfile
│   ├── xformers/Dockerfile
│   ├── torchao/Dockerfile
│   └── vllm/Dockerfile
├── docs/
│   ├── ngc_images.md         # NGC container inventory
│   ├── vllm.md               # vLLM architecture analysis
│   └── wheel_building.md     # Wheel building guide
└── wheels/                   # Build output (gitignored, released via GH Releases)
    ├── staging/
    ├── ngc-25.11-py3/
    ├── ngc-26.01-py3/
    └── pip-pt210-cu130/
```

## Related Projects

- **[vistralis/zimage-spark](https://github.com/vistralis/zimage-spark)** — ZImage diffusion pipeline (consumes wheels from this repo)
- **[vistralis/SageAttention](https://github.com/vistralis/SageAttention)** — Our SM 121 fork of SageAttention
