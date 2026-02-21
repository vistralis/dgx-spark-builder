# Seed Context for dgx-spark-builder

This document provides the full context needed to continue development on this project.
It is written for both human developers and AI coding agents.

## 1. Mission & Stack

Vistralis develops, trains, finetunes, optimizes, and deploys ML models for **physical AI** — robotics, autonomous vehicles, real-time analytics, and sensor-driven perception systems.

**Core stack**: PyTorch → NVIDIA ModelOpt → TensorRT → Triton Inference Server

This repo provides the build infrastructure to make that stack (and its dependencies) work on the DGX Spark's unique **aarch64 + Blackwell (SM 121)** platform, where most ML ecosystem packages lack prebuilt wheels.

Application images (ComfyUI, vLLM serving, diffusion pipelines, etc.) are downstream consumers — they use the wheels and base images built here, but are not the primary mission.

## 2. Hardware: DGX Spark GB10

The DGX Spark is NVIDIA's desktop Grace Blackwell system:

- **GPU**: GB10 — Blackwell architecture
  - Compute Capability: SM 12.0 (GB series) / SM 12.1 (production silicon)
  - CUDA arch flags: `TORCH_CUDA_ARCH_LIST="12.0;12.1"` or just `"12.1"`
- **CPU**: Grace (aarch64 / ARM64) — NOT x86_64
- **Memory**: 128 GB unified CPU+GPU memory
  - VMM (Virtual Memory Management): yes — allows addressing ~160 GB via oversubscription
- **CUDA**: 13.0 (native driver match with 25.11 NGC images)
- **Host OS**: Ubuntu 24.04
- **Python**: 3.12 (from NGC images or system)
- **TDP**: ~150W GPU, but inference typically draws ~45W

### Key Implications

1. **aarch64**: Most PyPI wheels are x86_64-only. CUDA extension packages must be built from source.
2. **SM 121**: Many CUDA kernels don't have Blackwell codepaths yet. `torch.compile` has a `KernelMetadata` crash on SM 121 (PyTorch bug, fixed in 2.11+).
3. **Unified memory**: No PCIe bottleneck. CPU and GPU share the same physical memory.
4. **CUDA 13.0**: Very new — most libraries haven't released builds for it.

## 3. Three ABI Targets

### ABI Compatibility Warning

NGC PyTorch images ship a **custom-built** PyTorch that differs at the C++ ABI level from pip-installed PyTorch. This means:
- Wheels built inside NGC containers → only work in NGC containers
- Wheels built with pip-installed PyTorch → only work with pip-installed PyTorch
- Mixing them causes: `"Cannot access data pointer of Tensor that doesn't have storage"`

### The Three Targets

| Output Dir | Base Image | PyTorch | Use Case |
|------------|-----------|---------|----------|
| `ngc-25.11-py3` | `nvcr.io/nvidia/pytorch:25.11-py3` | 2.10.0a0+nv25.11 | NGC-native workflows (ModelOpt, TRT) |
| `ngc-26.01-py3` | `nvcr.io/nvidia/pytorch:26.01-py3` | 2.10.0a0+nv26.01 | Latest NGC |
| `pip-pt210-cu130` | `ubuntu2404-pt210-cu130` (custom) | 2.10.0+cu130 (pip) | Lean pip-based apps (ComfyUI, etc.) |

## 4. Custom Base Image (`ubuntu2404-pt210-cu130`)

`dockerfiles/base/Dockerfile` creates a lean PyTorch environment:
- Starts from `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` (~9 GB)
- Installs Python 3.12, cmake, ninja, git
- Installs PyTorch from pip: `torch==2.10.0+cu130`
- Uses `uv` for fast pip operations
- Sets `TORCH_CUDA_ARCH_LIST="12.0;12.1"`
- Optionally builds PyTorch from source (`FORCE_BUILD=on`)

## 5. Build Infrastructure

### `artifacts.yaml`
Declares the build matrix: base images × packages. Key fields:
- `tag`: Docker image tag
- `output_dir`: Subdirectory under `wheels/`
- `local: true`: Image must be pre-built locally
- `enabled`: Whether to build by default

### `build_wheels.sh`
```bash
./build_wheels.sh                              # build all defaults
./build_wheels.sh sageattention                # build one
./build_wheels.sh --staging sageattention      # build to staging/
./build_wheels.sh --promote pip-pt210-cu130    # promote staging → target
BASE_IMAGE=ubuntu2404-pt210-cu130 ./build_wheels.sh onnxruntime
```

### `build_artifacts.py`
Python orchestrator reading `artifacts.yaml`. Supports:
- `--dry-run`, `--filter-image`, `--filter-package`, `--staging`

## 6. Package Build Details

| Package | Repo | Duration | Notes |
|---------|------|----------|-------|
| onnxruntime-gpu | microsoft/onnxruntime | ~30-45 min | Needs CUDA + TensorRT. NGC base has TRT; custom base does NOT. |
| bitsandbytes | bitsandbytes-foundation | ~2-3 min | `BNB_CUDA_VERSION=130` |
| sageattention | vistralis/SageAttention | ~5 min | Branch: `blackwell-sm121` |
| flash-attention | Dao-AILab/flash-attention | ~20-30 min | v2.8.3. ~2% slower than SDPA on GB10. |
| vllm | vllm-project/vllm | ~15-20 min | v0.16.0, `TORCH_CUDA_ARCH_LIST="12.0;12.1"` |
| xformers | facebookresearch/xformers | — | Disabled (SM 121 compile issues) |
| torchao | pytorch/ao | — | Disabled (SM 121 compile issues) |

## 7. Staging & Promote Workflow

```bash
# 1. Build → staging/
./build_wheels.sh --staging sageattention

# 2. Verify
docker run --gpus all -v $(pwd)/wheels/staging:/whl ubuntu2404-pt210-cu130 \
    python -c "import sageattention; print(sageattention.__version__)"

# 3. Promote → target dir (cleans Docker filesystem artifacts)
./build_wheels.sh --promote pip-pt210-cu130
```

## 8. Known Issues & Gotchas

1. **`torch.compile` crashes on SM 121**: `KernelMetadata` error. Fixed in PyTorch 2.11+ (PR #166185).
2. **NGC ABI mismatch**: Wheels built in NGC ≠ wheels for pip PyTorch.
3. **`docker build --output` filesystem leak**: Misconfigured `FROM scratch` can leak `.dockerenv`, `dev/`, `proc/` into output. Staging + promote workflow catches this.
4. **PyTorch SM 121 warning**: "GPU with compute capability 10.0 is not supported" — harmless.
5. **Cross-compilation (x86 → aarch64)**: QEMU emulation works but is 10-100× slower. Build natively on Spark.
6. **ORT on custom base**: No TensorRT dev libraries. Build without `--use_tensorrt` or add TRT to the base.

## 9. Future Work

- [ ] Re-enable xformers + torchao when SM 121 upstream fixes land
- [ ] Automate wheel publishing to GitHub Releases
- [ ] CI on aarch64 runner (GitHub Actions `ubuntu-24.04-arm`)
- [ ] ORT build variant: with/without TensorRT for custom base
- [ ] Triton Inference Server container definition
- [ ] ModelOpt optimization pipeline integration
