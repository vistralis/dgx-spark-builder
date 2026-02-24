# Docker Images

This project uses three types of Docker images:

## Base Images (`dockerfiles/images/`)

Foundation layers reused by builders and applications.

| Image | Tag | Contents |
|-------|-----|----------|
| **cuda-tensorrt** | `cuda13.0-tensorrt-ubuntu24.04` | CUDA 13.0 + cuDNN + TensorRT 10.14 + cmake, ninja, git, uv |
| **cuda-torch** | `cuda13.0-torch{VER}-ubuntu24.04` | + PyTorch from pip (parameterized version) |

```bash
# Build once, reused by everything else
docker build -t cuda13.0-tensorrt-ubuntu24.04 dockerfiles/images/cuda-tensorrt/

# Stable PyTorch
docker build --build-arg TORCH_VERSION=2.10.0 \
    -t cuda13.0-torch2.10-ubuntu24.04 dockerfiles/images/cuda-torch/

# RC PyTorch (bleeding edge)
docker build --build-arg TORCH_VERSION=2.11.0 \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/test/cu130 \
    -t cuda13.0-torch2.11rc1-ubuntu24.04 dockerfiles/images/cuda-torch/
```

## Builder Images (`dockerfiles/builders/`)

Compile wheels from source. Output is `.whl` files, extracted from a `FROM scratch` final stage.

| Builder | What it builds | Build time |
|---------|---------------|------------|
| `vllm` | vLLM serving engine | ~20 min |
| `sglang` | sgl-kernel + sglang | ~3 hours (MAX_JOBS=2) |
| `flash-attention` | FlashAttention 2/3 CUDA kernels | ~85 min |
| `flashinfer` | FlashInfer JIT Python package | ~3 sec |
| `torchao` | PyTorch AO quantization | ~2 min |
| `bitsandbytes` | NF4/INT8 quantization | ~3 min |
| `sageattention` | INT8/FP8 quantized attention | ~5 min |
| `onnxruntime` | ONNX Runtime GPU (CUDA + TRT) | ~45 min |
| `comfy-kitchen` | ComfyUI CUDA kernels | ~20 sec |
| `xformers` | Memory-efficient attention | 🔍 disabled (SM 121 issues) |

```bash
# Build a wheel (deposits to local directory)
docker build \
    --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    --output type=local,dest=wheels/torch2.10-cu130/ \
    dockerfiles/builders/flash-attention/

# Or build as intermediate image (extract later)
docker build \
    --build-arg BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 \
    -t flash-attn-builder:v2.8.3 \
    dockerfiles/builders/flash-attention/

# Extract wheels from builder image
cid=$(docker create flash-attn-builder:v2.8.3 -- true)
docker cp "$cid":/ /tmp/wheels/
docker rm "$cid"
```

## Application Images (`dockerfiles/images/`)

Deployable containers for running services.

| Image | Purpose |
|-------|---------|
| `comfyui` | ComfyUI node-based generation pipeline |
| `comfyui-qwen-tts` | ComfyUI + Qwen3-TTS voice synthesis + fine-tuning |

```bash
docker build -f dockerfiles/images/comfyui/Dockerfile .
```

## ABI Compatibility

> [!WARNING]
> **NGC PyTorch ≠ pip PyTorch**. NGC uses a custom C++ ABI. Mixing wheels built on different bases causes:
> `"Cannot access data pointer of Tensor that doesn't have storage"`

Each wheel release is tied to a specific base image. Do not mix across release tags.
