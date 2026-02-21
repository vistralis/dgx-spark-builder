# DGX Spark Builder

Build infrastructure for **NVIDIA DGX Spark GB10** (Blackwell, SM 121, aarch64) — providing custom CUDA wheel builds and Docker images for physical AI, analytics, robotics, and autonomous vehicle workloads.

## Mission

We develop, train, finetune, optimize, and deploy ML models for **physical AI** applications:
- Robotics perception and planning
- Autonomous vehicle (AV) analytics
- Real-time sensor fusion and inference
- Edge-to-cloud model deployment

Our core stack: **PyTorch → ModelOpt → TensorRT → Triton Inference Server**

This repo provides the build infrastructure to make that stack work on the DGX Spark's unique aarch64 + Blackwell (SM 121) platform, where most ML ecosystem packages lack prebuilt wheels.

## Hardware Target

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell) |
| Compute Capability | SM 12.0 / 12.1 |
| CPU Architecture | aarch64 (Grace) |
| CUDA | 13.0 / 13.1 |
| OS | Ubuntu 24.04 |
| Memory | 128 GB unified (VMM: ~160 GB addressable) |

## What This Repo Does

1. **Builds custom wheels** for packages with no prebuilt aarch64+CUDA wheels on PyPI
2. **Provides Dockerfiles** for application images (model serving, generation pipelines, etc.)
3. **Organizes wheels by ABI target** to ensure correct binary compatibility
4. **Publishes wheels via GitHub Releases** for easy consumption across projects

## Custom Wheels

| Package | Why Custom Build? |
|---------|-------------------|
| `onnxruntime-gpu` | No aarch64 CUDA wheels on PyPI |
| `bitsandbytes` | CPU-only aarch64 wheels, need CUDA build |
| `sageattention` | Our SM 121 fork with INT8/FP8 quantized attention |
| `flash-attn` | No aarch64 CUDA wheels on PyPI |
| `vllm` | No aarch64 build, need SM 120/121 support |
| `xformers` | No aarch64 wheels (currently disabled — SM 121 issues) |
| `torchao` | No aarch64 wheels (currently disabled — SM 121 issues) |

## Quick Start

```bash
# 1. Build the ABI-matched PyTorch base image
docker build -t ubuntu2404-pt210-cu130 dockerfiles/base/

# 2. Build wheels (to staging first)
BASE_IMAGE=ubuntu2404-pt210-cu130 ./build_wheels.sh --staging sageattention bitsandbytes

# 3. Verify, then promote
./build_wheels.sh --promote pip-pt210-cu130

# 4. Build an application image
docker build -f dockerfiles/comfyui/Dockerfile \
    --build-context wheels=wheels/pip-pt210-cu130 .
```

## Three Base Image Targets

| Target | Tag | Use Case |
|--------|-----|----------|
| **NGC PyTorch 25.11** | `nvcr.io/nvidia/pytorch:25.11-py3` | Matches host driver, batteries-included |
| **NGC PyTorch 26.01** | `nvcr.io/nvidia/pytorch:26.01-py3` | Latest NGC, forward-compat drivers |
| **Custom pip base** | `ubuntu2404-pt210-cu130` | Lean, ABI-clean pip-installed PyTorch |

> [!WARNING]
> Wheels built on NGC images are **NOT** compatible with the custom pip base
> and vice versa. NGC ships a custom PyTorch build with different C++ ABI.

## Project Structure

```
dgx-spark-builder/
├── README.md                 # This file
├── CONTEXT.md                # Seed context for AI agents
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
│   └── wheel_building.md     # Wheel building guide
└── wheels/                   # Build output (gitignored, published as GH Releases)
```

## Related Projects

- **[vistralis/zimage-spark](https://github.com/vistralis/zimage-spark)** — Diffusion pipeline (consumes wheels from this repo)
- **[vistralis/SageAttention](https://github.com/vistralis/SageAttention)** — SM 121 fork of SageAttention
