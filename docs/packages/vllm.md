# vLLM

| | |
|---|---|
| **Repo** | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **Version** | 0.16.0 |
| **Status** | ✅ Published |
| **Wheel** | `vllm-0.16.0+cu130-cp312-cp312-linux_aarch64.whl` (~504-553 MB) |

## What it does

High-throughput LLM serving engine with PagedAttention, continuous batching, and OpenAI-compatible API. Supports 100+ model architectures, speculative decoding, and quantization (AWQ, GPTQ, FP8, INT8).

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ Supported via `TORCH_CUDA_ARCH_LIST="12.0;12.1"` |
| SM 121 | ✅ with patch — `dsv3_fused_a_gemm` guard required |
| aarch64 | ✅ Builds from source |
| torch 2.10 | ✅ |
| torch 2.11 | ✅ via [atalman/update_torch_211](https://github.com/vllm-project/vllm/pull/34644) PR |

## Patches Required

The `dsv3_fused_a_gemm` kernel (DeepSeek V3 fused GEMM) only compiles for SM 90+. On SM 12.x, the kernel is skipped but the binding is still registered, causing an undefined symbol error at import. Our [vistralis/vllm](https://github.com/vistralis/vllm/tree/fix/dsv3-sm120-torch211) fork wraps it with `#ifdef ENABLE_DSV3_FUSED_A_GEMM`.

## Build

```bash
docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    --build-arg VLLM_VERSION=v0.16.0 \
    -t vllm-builder:v0.16.0 \
    dockerfiles/builders/vllm/
```

Build time: ~20 minutes.
