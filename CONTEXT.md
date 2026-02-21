# Seed Context for dgx-spark-builder

This document provides the full context needed to continue development on this project.
It is written for both human developers and AI coding agents.

## 1. Hardware: DGX Spark GB10

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

## 2. NGC Base Images

We use NVIDIA NGC container images as our starting point. Key image inventory:

| Image | Tag | CUDA | Size | PyTorch | Use Case |
|-------|-----|------|------|---------|----------|
| pytorch | 25.11-py3 | 13.0 | 19.5 GB | 2.10.0a0 | Native driver match |
| pytorch | 26.01-py3 | 13.1 | 19.9 GB | 2.10.0a0 | Latest NGC |
| tensorrt | 26.01-py3 | 13.1 | 10.8 GB | ❌ | TRT-only inference |
| cuda-dl-base | 26.01-inference-devel | 13.1 | 5.2 GB | ❌ | Smallest devel base |
| vllm | 26.01-py3 | 13.1 | — | ✅ | LLM serving |

### ABI Compatibility Warning

NGC PyTorch images ship a **custom-built** PyTorch that differs at the C++ ABI level from pip-installed PyTorch. This means:
- Wheels built inside NGC containers → only work in NGC containers
- Wheels built with pip-installed PyTorch → only work with pip-installed PyTorch
- Mixing them causes: `"Cannot access data pointer of Tensor that doesn't have storage"`

This is why we have **three separate wheel output directories** — one per ABI target.

## 3. Custom Base Image (`ubuntu2404-pt210-cu130`)

Our `dockerfiles/base/Dockerfile` creates a lean PyTorch environment:
- Starts from `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` (~9 GB)
- Installs Python 3.12, cmake, ninja, git
- Installs PyTorch from pip: `torch==2.10.0+cu130` from `download.pytorch.org/whl/cu130`
- Uses `uv` for fast pip operations
- Sets `TORCH_CUDA_ARCH_LIST="12.0;12.1"`
- Optionally builds PyTorch from source (`FORCE_BUILD=on`) for exact ABI match

This image is the base for all custom application images (ComfyUI, etc.) where we want a lean, ABI-clean environment without NGC overhead.

## 4. Build Infrastructure

### `artifacts.yaml`

Declares the build matrix: which base images × which packages. Each entry has:
- `tag`: Docker image tag (full URI for NGC, local name for custom base)
- `output_dir`: Name of the wheels output subdirectory
- `local`: true if the image must be built locally first
- `enabled`: whether to build by default

### `build_wheels.sh`

Bash script that builds one or more wheels:
```bash
./build_wheels.sh                              # build all defaults
./build_wheels.sh sageattention flash-attention # build specific
./build_wheels.sh --staging sageattention       # build to staging/
./build_wheels.sh --promote pip-pt210-cu130     # move staging → target
BASE_IMAGE=ubuntu2404-pt210-cu130 ./build_wheels.sh onnxruntime  # override base
```

All builds use `docker build --output type=local,dest=<dir>` to extract only the `.whl` files from multi-stage Docker builds (final stage is `FROM scratch`).

### `build_artifacts.py`

Python orchestrator that reads `artifacts.yaml` and runs builds sequentially. Supports:
- `--dry-run`: show build plan without building
- `--filter-image <glob>`: build only matching images
- `--filter-package <glob>`: build only matching packages
- `--package <name>`: build exactly one package (overrides enabled flag)

### `probe_images.sh`

Probes NGC images for installed packages. Outputs JSON per image with:
- Python version, CUDA version, image size
- Installed packages: torch, tensorrt, onnxruntime, vllm, transformers, etc.

## 5. Package Build Details

### onnxruntime-gpu
- **Repo**: microsoft/onnxruntime (or fork)
- **Build**: `./build.sh --use_cuda --use_tensorrt --build_wheel`
- **Duration**: ~30-45 min on Spark
- **Output**: ~54 MB wheel
- **Note**: Requires CUDA + TensorRT dev libraries in base image. NGC images have TensorRT; the custom base does NOT unless added.

### bitsandbytes
- **Repo**: bitsandbytes-foundation/bitsandbytes
- **Build**: `pip wheel . --no-build-isolation --no-deps`
- **Duration**: ~2-3 min
- **Env vars**: `BNB_CUDA_VERSION=130`, `TORCH_CUDA_ARCH_LIST="12.1"`

### sageattention
- **Repo**: vistralis/SageAttention (our fork with SM 121 support)
- **Branch**: `blackwell-sm121`
- **Build**: `pip wheel . --no-build-isolation --no-deps`
- **Note**: Uses the custom base image by default (ABI-matched to pip PyTorch)

### flash-attention
- **Repo**: Dao-AILab/flash-attention
- **Version**: v2.8.3
- **Build**: `pip wheel . --no-deps --no-build-isolation`
- **Duration**: ~20-30 min
- **Note**: FA2 is ~2% SLOWER than PyTorch SDPA on GB10 (SM 121). Provided for compatibility. FA3/FA4 Blackwell support pending.

### xformers (disabled)
- **Repo**: facebookresearch/xformers
- **Status**: SM 121 compile issues. Disabled until upstream fixes.

### torchao (disabled)
- **Repo**: pytorch/ao
- **Status**: SM 121 compile issues. Disabled until upstream fixes.

### vllm
- **Repo**: vllm-project/vllm
- **Version**: v0.16.0
- **Build**: `python setup.py bdist_wheel`
- **Key**: `TORCH_CUDA_ARCH_LIST="12.0;12.1"`, `VLLM_VERSION_OVERRIDE="0.16.0+cu130"`
- **Note**: v0.16.0 pins torch==2.10.0 which matches our base. Earlier versions (v0.15.1) pinned torch==2.9.1 and needed filtering.

## 6. Application Images

### ComfyUI (`dockerfiles/comfyui/Dockerfile`)
- Base: `ubuntu2404-pt210-cu130`
- Installs: torchaudio, torchvision from PyTorch cu130 index
- Installs: ComfyUI from GitHub
- Copies: sageattention, bitsandbytes, flash-attn wheels (by name, not glob)
- System deps: libsndfile1, sox, libglfw3-dev, libgl1-mesa-dev
- Port: 8188

### ZImage Pipeline (lives in vistralis/zimage-spark)
- Base: `nvcr.io/nvidia/pytorch:26.01-py3` (NGC, for diffusers + TensorRT)
- Consumes: onnxruntime-gpu wheel from this repo
- Separate project, not built here

## 7. Known Issues & Gotchas

1. **`torch.compile` crashes on SM 121**: `KernelMetadata` error. Fixed in PyTorch 2.11+ (PR #166185). Workaround: don't use `torch.compile` or `torch_tensorrt`.

2. **NGC ABI mismatch**: Wheels built in NGC containers CANNOT be used with pip-installed PyTorch and vice versa. The error is: `"Cannot access data pointer of Tensor that doesn't have storage"`.

3. **`docker build --output` leaks container filesystem**: If `FROM scratch` + `COPY` is misconfigured, you get `.dockerenv`, `dev/`, `proc/`, `sys/` in the output. Always verify the output directory only contains `.whl` files.

4. **GPU power on Spark**: ~45W during inference vs ~150W TDP. Clock sustains at 2418 MHz (max 3003 MHz).

5. **PyTorch SM 121 warning**: "GPU with compute capability 10.0 is not supported" — harmless, fixed in 2.11+.

6. **Cross-compilation (x86 → aarch64)**: Docker BuildKit + QEMU works but is 10-100× slower for CUDA extension compilation. Not practical for heavy builds. Build natively on the Spark.

7. **`onnxruntime` without TensorRT**: The custom base image (`nvidia/cuda:*`) does NOT include TensorRT. To build ORT with TensorRT support, either:
   - Use an NGC base (which includes TensorRT)
   - Install TensorRT dev packages in the custom base
   - Build ORT without `--use_tensorrt` (loses TRT execution provider)

## 8. Staging & Promote Workflow

To avoid accidental overwrites and ensure wheel quality:

```bash
# 1. Build → staging/
./build_wheels.sh --staging sageattention

# 2. Verify
docker run --gpus all -v $(pwd)/wheels/staging:/whl ubuntu2404-pt210-cu130 \
    python -c "import sageattention; print(sageattention.__version__)"

# 3. Promote → target dir
./build_wheels.sh --promote pip-pt210-cu130
```

Staging is cleaned on promote. Only `.whl` files are moved (Docker filesystem artifacts are discarded).

## 9. GitHub Releases for Wheels

Wheels are published as GitHub Release assets for easy download:

```bash
# Download a specific wheel
gh release download v0.1.0 -p 'sageattention-*.whl' -D wheels/pip-pt210-cu130/

# Or via URL
curl -LO https://github.com/vistralis/dgx-spark-builder/releases/download/v0.1.0/sageattention-2.2.0-cp312-cp312-linux_aarch64.whl
```

## 10. Future Work

- [ ] ComfyUI Dockerfile (in progress)
- [ ] Add ComfyUI-Qwen-TTS node support
- [ ] Re-enable xformers + torchao when SM 121 upstream fixes land
- [ ] Automate wheel publishing to GitHub Releases
- [ ] CI on aarch64 runner (GitHub Actions ubuntu-24.04-arm)
- [ ] ORT build option: with/without TensorRT for custom base
