# Seed Context for dgx-spark-builder

This document provides the full context needed to continue development on this project.
It is written for both human developers and AI coding agents.

## 1. Mission & Stack

Vistralis develops, trains, finetunes, optimizes, and deploys ML models for **physical AI** — robotics, autonomous vehicles, real-time analytics, and sensor-driven perception systems.

**Core stack**: PyTorch → NVIDIA ModelOpt → TensorRT → Triton Inference Server

This repo provides the build infrastructure to make that stack (and its dependencies) work on the DGX Spark’s unique **aarch64 + Blackwell (SM 121)** platform, where most ML ecosystem packages lack prebuilt wheels.

Application images (ComfyUI, vLLM serving, diffusion pipelines, etc.) are downstream consumers — they use the wheels and base images built here, but are not the primary mission.

## 2. Hardware: DGX Spark GB10

- **GPU**: GB10 — Blackwell, SM 12.0/12.1
  - CUDA arch flags: `TORCH_CUDA_ARCH_LIST="12.0;12.1"`
- **CPU**: Grace (aarch64 / ARM64) — NOT x86_64
- **Memory**: 128 GB unified (VMM: ~160 GB addressable)
- **CUDA**: 13.0 (native driver), 13.1 (forward-compat)
- **Host OS**: Ubuntu 24.04, Python 3.12
- **Power**: ~45W GPU during inference, ~150W TDP

### Key Implications

1. **aarch64**: Most PyPI wheels are x86_64-only. CUDA extensions must be built from source.
2. **SM 121**: `torch.compile` has a `KernelMetadata` crash (PyTorch bug, fixed in 2.11+).
3. **Unified memory**: No PCIe bottleneck. CPU/GPU share physical memory.
4. **CUDA 13.0**: Very new — most libraries lack builds for it.

## 3. Three ABI Targets

NGC PyTorch uses a custom C++ ABI. Mixing NGC-built wheels with pip-PyTorch causes:
`"Cannot access data pointer of Tensor that doesn't have storage"`

| Output Dir | Base Image | PyTorch |
|------------|-----------|----------|
| `torch2.10-cu130` | `cuda13.0-torch2.10-ubuntu24.04` | 2.10.0+cu130 (pip) |
| `torch2.11rc1-cu130` | `cuda13.0-torch2.11rc1-ubuntu24.04` | 2.11.0rc1+cu130 (pip) |

## 4. Custom Base Image (`cuda13.0-torch2.10-ubuntu24.04`)

`dockerfiles/base/Dockerfile`:
- From `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` (~9 GB)
- Python 3.12, cmake, ninja, git, `uv`
- PyTorch from pip: `torch==2.10.0+cu130`
- Optional from-source build: `FORCE_BUILD=on`

## 5. Build Infrastructure

### `artifacts.yaml`
Build matrix: base images × packages. Fields: `tag`, `output_dir`, `local`, `enabled`.

### `build_wheels.sh`
```bash
./build_wheels.sh                              # build all defaults → wheels/torch2.10-cu130/
./build_wheels.sh sageattention flash-attention # build specific targets
BASE_IMAGE=cuda13.0-torch2.11rc1-ubuntu24.04 ./build_wheels.sh vllm  # → wheels/torch2.11rc1-cu130/
```

### `build_artifacts.py`
Python orchestrator: `--dry-run`, `--filter-image`, `--filter-package`.

## 6. Package Build Details

| Package | Repo | Duration | Notes |
|---------|------|----------|-------|
| onnxruntime-gpu | microsoft/onnxruntime | ~30-45 min | Needs TensorRT (NGC has it, custom base doesn't) |
| bitsandbytes | bitsandbytes-foundation | ~2-3 min | `BNB_CUDA_VERSION=130` |
| sageattention | vistralis/SageAttention | ~5 min | Branch: `blackwell-sm121` |
| flash-attention | Dao-AILab/flash-attention | ~20-30 min | ~2% slower than SDPA on GB10 |
| vllm | vllm-project/vllm | ~15-20 min | v0.16.0, SM 12.0/12.1 |
| xformers | facebookresearch/xformers | — | Disabled (SM 121 issues) |
| torchao | pytorch/ao | ~2 min | v0.16.0 + commit 6ad7c404 |
| comfy-kitchen | Comfy-Org/comfy-kitchen | ~20s | CUDA + Triton backends |
| flashinfer | flashinfer-ai/flashinfer | ~3s | v0.6.4 JIT (pure-Python) |

## 7. Known Issues

1. **`torch.compile`** crashes on SM 121 (fixed in PyTorch 2.11+)
2. **ABI mismatch**: NGC wheels ≠ pip wheels
3. **`docker build --output`** can leak container filesystem artifacts
4. **ORT on custom base**: No TensorRT — build without `--use_tensorrt` or add TRT
5. **Cross-compilation** (x86→aarch64): QEMU works but 10-100× slower

## 8. Future Work

- [ ] Re-enable xformers + torchao when SM 121 upstream fixes land
- [ ] Automate wheel publishing to GitHub Releases
- [ ] CI on aarch64 runner (GitHub Actions `ubuntu-24.04-arm`)
- [ ] ORT build variant: with/without TensorRT
- [ ] Triton Inference Server container definition
- [ ] ModelOpt optimization pipeline integration
