# SageAttention

| | |
|---|---|
| **Repo** | [vistralis/SageAttention](https://github.com/vistralis/SageAttention) (SM 121 fork) |
| **Upstream** | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) |
| **Version** | 2.2.0 |
| **Status** | ✅ Published |
| **Wheel** | `sageattention-2.2.0-cp312-cp312-linux_aarch64.whl` (15 MB) |

## What it does

Integer-quantized attention mechanism. INT8/FP8 quantized Q/K computation for memory-efficient attention with near-lossless quality. 2-3x speedup over standard attention.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ |
| SM 121 | ✅ via our fork (`blackwell-sm121` branch) |
| aarch64 | ✅ |
| torch 2.10 | ✅ |
| torch 2.11 | ✅ |

Our fork adds SM 121 support via the `blackwell-sm121` branch with Triton kernel adjustments.

## Build

```bash
docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    dockerfiles/builders/sageattention/
```

Build time: ~5 minutes.
