# torchao

| | |
|---|---|
| **Repo** | [pytorch/ao](https://github.com/pytorch/ao) |
| **Version** | 0.16.0 |
| **Status** | ✅ Published |
| **Wheel** | `torchao-0.16.0+git*-cp310-abi3-linux_aarch64.whl` (4 MB) |

## What it does

PyTorch native quantization and sparsity library. FP8, INT8, INT4, and NF4 quantization for inference and training. Used by vLLM for some quantization methods.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ |
| SM 121 | ✅ (abi3, stable ABI) |
| aarch64 | ✅ |
| torch 2.10 | ✅ |
| torch 2.11 | ✅ |

## Build

```bash
docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    dockerfiles/builders/torchao/
```

Build time: ~2 minutes.
