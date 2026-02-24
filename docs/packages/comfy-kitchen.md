# comfy-kitchen

| | |
|---|---|
| **Repo** | [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) |
| **Version** | 0.2.7 |
| **Status** | ✅ Published |
| **Wheel** | `comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl` (659 KB) |

## What it does

CUDA kernel library for ComfyUI. Provides optimized GPU operations for the ComfyUI image generation pipeline with CUDA, Triton, and eager backends.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ |
| SM 121 | ✅ (SM 12.0/12.1 supported) |
| aarch64 | ✅ |
| torch 2.10 | ✅ |
| torch 2.11 | ✅ |

## Build

```bash
docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    dockerfiles/builders/comfy-kitchen/
```

Build time: ~20 seconds.
