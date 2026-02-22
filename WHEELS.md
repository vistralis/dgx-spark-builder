# Wheel Inventory

Pre-built aarch64 wheels for **DGX Spark GB10** (Blackwell, SM 12.0/12.1, CUDA 13.0).
Published as [GitHub Releases](https://github.com/vistralis/dgx-spark-builder/releases).

> [!IMPORTANT]
> These wheels are built against our custom pip-based base images (`cuda13.0-torch*-ubuntu24.04`).
> They are **NOT** compatible with NGC container PyTorch builds (different C++ ABI).

---

## [`torch2.10-cu130`](https://github.com/vistralis/dgx-spark-builder/releases/tag/torch2.10-cu130)

Base image: `cuda13.0-torch2.10-ubuntu24.04` (PyTorch 2.10.0 stable)

| Wheel | Size | Source | Notes |
|-------|------|--------|-------|
| `vllm-0.16.0+cu130-cp312-cp312-linux_aarch64.whl` | 553 MB | [vllm-project/vllm@v0.16.0](https://github.com/vllm-project/vllm/tree/v0.16.0) | FA2+FA3, CUTLASS MLA, Marlin, MoE kernels |
| `flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl` | 232 MB | [Dao-AILab/flash-attention@v2.8.3](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3) | SM 12.0 support via CUTLASS |
| `flashinfer_python-0.6.4-py3-none-any.whl` | 7.5 MB | [flashinfer-ai/flashinfer@v0.6.4](https://github.com/flashinfer-ai/flashinfer/tree/v0.6.4) | Pure-Python JIT; compiles CUDA kernels at runtime via CUTLASS |
| `comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl` | 659 KB | [Comfy-Org/comfy-kitchen@v0.2.7](https://github.com/Comfy-Org/comfy-kitchen/tree/v0.2.7) | CUDA + Triton + eager backends, SM 12.0/12.1 |
| `onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl` | 55 MB | [microsoft/onnxruntime@v1.25.0](https://github.com/microsoft/onnxruntime/tree/v1.25.0) | CUDA EP + TensorRT EP |
| `sageattention-2.2.0-cp312-cp312-linux_aarch64.whl` | 15 MB | [vistralis/SageAttention](https://github.com/vistralis/SageAttention) | Our SM 121 fork, INT8/FP8 quantized attention |
| `torchao-0.16.0+git3c1065ca6-cp310-abi3-linux_aarch64.whl` | 4 MB | [pytorch/ao@v0.16.0](https://github.com/pytorch/ao/tree/v0.16.0) (commit `3c1065ca`) | Stable ABI (cp310+) |
| `torchao-0.16.0+git6ad7c4046-cp310-abi3-linux_aarch64.whl` | 4 MB | [pytorch/ao@6ad7c404](https://github.com/pytorch/ao/commit/6ad7c40461a5c8e79e442f17d8c25b47f0ae652f) | Same as flux env version |

---

## [`torch2.11rc1-cu130`](https://github.com/vistralis/dgx-spark-builder/releases/tag/torch2.11rc1-cu130)

Base image: `cuda13.0-torch2.11rc1-ubuntu24.04` (PyTorch 2.11.0 RC1 from test index)

| Wheel | Size | Source | Notes |
|-------|------|--------|-------|
| `vllm-0.16.0+cu130-cp312-cp312-linux_aarch64.whl` | 504 MB | [atalman/vllm@0b99aaab](https://github.com/atalman/vllm/commit/0b99aaab5bf26be56a30681a29cc5894b975a9e3) ([PR #34644](https://github.com/vllm-project/vllm/pull/34644)) | torch 2.11 compatibility patches; FA2+FA3 |
| `flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl` | 232 MB | [Dao-AILab/flash-attention@v2.8.3](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3) | SM 80/90/100/120, ~85 min build |
| `onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl` | 55 MB | [microsoft/onnxruntime@v1.25.0](https://github.com/microsoft/onnxruntime/tree/v1.25.0) | CUDA EP + TensorRT EP |
| `sageattention-2.2.0-cp312-cp312-linux_aarch64.whl` | 15 MB | [vistralis/SageAttention](https://github.com/vistralis/SageAttention) | Our SM 121 fork, INT8/FP8 quantized attention |
| `torchao-0.16.0+git6ad7c4046-cp310-abi3-linux_aarch64.whl` | 4 MB | [pytorch/ao@6ad7c404](https://github.com/pytorch/ao/commit/6ad7c40461a5c8e79e442f17d8c25b47f0ae652f) | Stable ABI (cp310+) |
| `bitsandbytes-0.50.0.dev0-cp312-cp312-linux_aarch64.whl` | 1.3 MB | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | CUDA 13.0 aarch64 |
| `comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl` | 659 KB | [Comfy-Org/comfy-kitchen@v0.2.7](https://github.com/Comfy-Org/comfy-kitchen/tree/v0.2.7) | CUDA + Triton + eager backends, SM 12.0/12.1 |

> [!NOTE]
> The torch 2.11 vLLM wheel was built from [PR #34644](https://github.com/vllm-project/vllm/pull/34644)
> (`atalman:update_torch_211`), which patches vLLM main with `torch==2.11.0` version pins.
> This PR is still open upstream — the patches are minimal (version bumps + one CPU-only API change).

---

## Build Reproducibility

All wheels are built via Dockerfiles in `dockerfiles/`. Example:

```bash
# vLLM (torch 2.10, stable)
docker build --no-cache \
    --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    --build-arg VLLM_VERSION=v0.16.0 \
    -t vllm-builder:v0.16.0 \
    dockerfiles/vllm/

# vLLM (torch 2.11, from PR)
docker build --no-cache \
    --build-arg BASE_IMAGE=cuda13.0-torch2.11rc1-ubuntu24.04 \
    --build-arg VLLM_REPO=https://github.com/atalman/vllm.git \
    --build-arg VLLM_VERSION=0b99aaab5bf26be56a30681a29cc5894b975a9e3 \
    -t vllm-builder:torch2.11 \
    dockerfiles/vllm/
```

Extract wheels from any builder image:

```bash
cid=$(docker create <image>) && docker cp "$cid":/ /tmp/wheels && docker rm "$cid"
```

## Installation

```bash
pip install --find-links https://github.com/vistralis/dgx-spark-builder/releases/download/torch2.10-cu130/ \
    vllm flash-attn sageattention torchao onnxruntime-gpu
```
