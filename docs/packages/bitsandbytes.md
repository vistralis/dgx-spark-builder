# bitsandbytes

| | |
|---|---|
| **Repo** | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| **Version** | 0.50.0.dev0 |
| **Status** | ✅ Published |
| **Wheel** | `bitsandbytes-0.50.0.dev0-cp312-cp312-linux_aarch64.whl` (1.3 MB) |

## What it does

8-bit and 4-bit quantization library. Enables loading large models in NF4/INT8 format with minimal quality loss. Used by HuggingFace Transformers for `load_in_4bit` / `load_in_8bit`.

## DGX Spark Support

| Aspect | Status |
|--------|--------|
| CUDA 13.0 | ✅ — `BNB_CUDA_VERSION=130` |
| SM 121 | ✅ |
| aarch64 | ✅ (PyPI only has CPU wheels for aarch64) |
| torch 2.10 | ✅ |
| torch 2.11 | ✅ |

## Build

```bash
docker build --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    dockerfiles/builders/bitsandbytes/
```

Build time: ~3 minutes.
