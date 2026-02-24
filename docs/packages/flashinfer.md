# FlashInfer

| | |
|---|---|
| **Repo** | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) |
| **Version** | 0.6.4 |
| **Status** | ✅ Published |
| **Wheel** | `flashinfer_python-0.6.4-py3-none-any.whl` (7.5 MB) |

## What it does

Kernel library for LLM serving — attention, decode, prefill, page table operations. Pure-Python package that JIT-compiles CUDA kernels at runtime via CUTLASS. Default attention backend for SGLang.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ |
| SM 121 | ✅ — `FLASHINFER_ENABLE_SM120=1` |
| aarch64 | ✅ |
| torch 2.10 | ✅ (JIT compiles against installed torch) |
| torch 2.11 | ✅ |

Pure Python wheel — no ABI lock-in. JIT compilation uses the host's CUDA toolkit and torch at runtime.

## Build

```bash
docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    dockerfiles/builders/flashinfer/
```

Build time: ~3 seconds (Python-only packaging).
