# DGX Spark Builder

Build infrastructure for **NVIDIA DGX Spark GB10** — custom CUDA wheels and Docker images for the aarch64 + Blackwell (SM 121) platform.

## Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell, SM 12.0 / 12.1) |
| CPU | Grace (aarch64) |
| CUDA | 13.0 / 13.1 |
| Memory | 128 GB unified (VMM: ~160 GB) |
| OS | Ubuntu 24.04, Python 3.12 |

Most ML packages lack prebuilt aarch64 + CUDA 13.0 wheels. This repo builds them from source.

## Wheel Releases

Pre-built wheels published as [GitHub Releases](https://github.com/vistralis/dgx-spark-builder/releases):

| Release | PyTorch | Packages |
|---------|---------|----------|
| [`torch2.10-cu130`](https://github.com/vistralis/dgx-spark-builder/releases/tag/torch2.10-cu130) | 2.10.0 (stable) | vllm, sgl-kernel, sglang, flash-attn, flashinfer, torchao, bitsandbytes, sageattention, onnxruntime-gpu, comfy-kitchen |
| [`torch2.11rc1-cu130`](https://github.com/vistralis/dgx-spark-builder/releases/tag/torch2.11rc1-cu130) | 2.11.0 RC1 | vllm, sgl-kernel, sglang, flash-attn, flashinfer, torchao, bitsandbytes, sageattention, onnxruntime-gpu, comfy-kitchen |

> [!IMPORTANT]
> Wheels are NOT compatible with NGC container PyTorch (different C++ ABI).
> See [docs/images.md](docs/images.md) for details.

📦 [Full wheel inventory](docs/wheels.md) · 📋 [Package support analysis](docs/packages/)

## Quick Start

```bash
# 1. Build base images
docker build -t cuda13.0-tensorrt-ubuntu24.04 dockerfiles/images/cuda-tensorrt/
docker build --build-arg TORCH_VERSION=2.10.0 \
    -t cuda13.0-torch2.10-ubuntu24.04 dockerfiles/images/cuda-torch/

# 2. Build a wheel
docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    --output type=local,dest=wheels/torch2.10-cu130/ \
    dockerfiles/builders/flash-attention/

# 3. Or use the build script
scripts/build_wheels.sh flash-attention sageattention

# 4. Install from release
pip install --find-links https://github.com/vistralis/dgx-spark-builder/releases/download/torch2.10-cu130/ \
    vllm flash-attn sageattention
```

## Repo Structure

```
dockerfiles/
├── images/                    # Base + application images
│   ├── cuda-tensorrt/         #   CUDA 13.0 + TensorRT + build tools
│   ├── cuda-torch/            #   + PyTorch (parameterized version)
│   ├── comfyui/               #   ComfyUI application
│   └── comfyui-qwen-tts/      #   ComfyUI + Qwen3-TTS voice synthesis
│
└── builders/                  # Wheel builders (output = .whl files)
    ├── vllm/                  sglang/              flash-attention/
    ├── flashinfer/            torchao/             bitsandbytes/
    ├── sageattention/         onnxruntime/         comfy-kitchen/
    └── xformers/

scripts/                       # build_wheels.sh, build_artifacts.py, probe_images.sh
docs/                          # Detailed documentation
```

## Documentation

| Doc | Contents |
|-----|----------|
| [wheels.md](docs/wheels.md) | Full wheel inventory per release with sources and sizes |
| [images.md](docs/images.md) | Docker image types: base vs builder vs application |
| [packages/](docs/packages/) | Per-package DGX Spark support analysis and build notes |
| [building.md](docs/building.md) | How to build wheels from source |
| [ngc_images.md](docs/ngc_images.md) | NGC container reference |

## Related

- [vistralis/SageAttention](https://github.com/vistralis/SageAttention) — SM 121 fork
- [vistralis/vllm](https://github.com/vistralis/vllm) — dsv3 SM 12.x fix
