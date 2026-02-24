# SGLang

| | |
|---|---|
| **Repo** | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| **Version** | 0.5.9 (sgl-kernel 0.3.21) |
| **Status** | ✅ Published |
| **Wheels** | `sgl_kernel-0.3.21-cp310-abi3-linux_aarch64.whl` (232 MB) + `sglang-0.5.9-py3-none-any.whl` (5 MB) |

## What it does

LLM serving engine with RadixAttention (automatic prefix caching), structured generation, and an OpenAI-compatible API. Ships as two packages: `sgl-kernel` (compiled CUDA kernels) and `sglang` (pure Python serving framework).

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ Native support — CMake auto-detects SM 103a, 110a, 121a |
| SM 121 | ✅ No patches needed |
| aarch64 | ✅ Builds from source |
| torch 2.10 | ✅ (separate sgl-kernel build required — libtorch ABI) |
| torch 2.11 | ✅ |

## Architecture

- **sgl-kernel**: Two compiled `.so` libraries — `sm90/` (fast math) and `sm100/` (precise math for SM 100+/121)
- **sglang**: Pure Python — framework, scheduler, API server
- The `abi3` tag means Python 3.10+ stable ABI, but **NOT** cross-torch-version — linked against specific libtorch

## Build Notes

sgl-kernel compiles FA3, FA4, FlashMLA, DeepGEMM, MoE kernels, and custom attention across SM 90/90a/100a/103a/110a/121a. This is extremely memory-intensive:

| MAX_JOBS | Result on 128 GB |
|----------|-----------------|
| 16 | OOM (killed) |
| 8 | OOM (killed) |
| 4 | OOM (killed) |
| **2** | ✅ Success (~2h 47m) |

System deps required: `libnuma-dev`, `libibverbs-dev`
CMake workaround: `CMAKE_POLICY_VERSION_MINIMUM=3.5` (dlpack fetch)

## Build

```bash
docker build --no-cache \
    --build-arg BASE_IMAGE=cuda13.0-torch2.11rc1-ubuntu24.04 \
    --build-arg SGLANG_VERSION=v0.5.9 \
    -t sglang-builder:v0.5.9-torch211 \
    dockerfiles/builders/sglang/
```

Build time: ~3 hours at MAX_JOBS=2.
