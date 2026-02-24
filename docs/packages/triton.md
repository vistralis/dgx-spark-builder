# Triton

| | |
|---|---|
| **Repo** | [triton-lang/triton](https://github.com/triton-lang/triton) |
| **Version** | 3.3.x |
| **Status** | 🔍 Evaluating |

## What it does

GPU programming language and compiler for writing high-performance CUDA kernels in Python. Used by PyTorch (`torch.compile`), FlashInfer, and many custom kernel libraries.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ PyTorch bundles a compatible version |
| SM 121 | ⚠️ `torch.compile` crashes on PyTorch < 2.11 |
| aarch64 | ✅ (bundled with PyTorch) |
| torch 2.10 | ✅ (bundled triton 3.2.x) |
| torch 2.11 | ✅ (bundled triton 3.3.x, fixes SM 121 crash) |

## Considerations

- PyTorch **already bundles Triton** — a custom build is only needed if:
  - The bundled version has SM 121-specific bugs
  - A newer Triton version adds Blackwell-specific optimizations
- Building Triton from source is complex (LLVM dependency)
- Low priority unless specific Triton bugs are encountered
