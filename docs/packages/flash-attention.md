# FlashAttention

| | |
|---|---|
| **Repo** | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| **Version** | 2.8.3 |
| **Status** | ✅ Published |
| **Wheel** | `flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl` (232 MB) |

## What it does

IO-aware exact attention implementation. Includes FlashAttention-2 (CUDA) and FlashAttention-3 (Hopper+). Used by vLLM, SGLang, and many training pipelines.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ |
| SM 121 | ✅ via CUTLASS path |
| aarch64 | ✅ |
| torch 2.10 | ✅ |
| torch 2.11 | ✅ |

No patches required. SM 12.0 uses the CUTLASS-based implementation path.

## Build

```bash
docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    dockerfiles/builders/flash-attention/
```

Build time: ~85 minutes. Builds SM 80/90/100/120 arch variants.
