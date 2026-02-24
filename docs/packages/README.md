# Package Support

DGX Spark compatibility analysis for ML ecosystem packages. Each package page covers: what it does, upstream status, CUDA 13.0 / SM 121 support, PyTorch 2.10/2.11 compatibility, and build notes.

## Supported Packages

Wheels built and published in [GitHub Releases](https://github.com/vistralis/dgx-spark-builder/releases).

| Package | Version | Repo | torch 2.10 | torch 2.11 | Build Time | Notes |
|---------|---------|------|:----------:|:----------:|-----------|-------|
| [vLLM](vllm.md) | 0.16.0 | [vllm-project/vllm](https://github.com/vllm-project/vllm) | ✅ | ✅¹ | ~20 min | LLM serving engine |
| [SGLang](sglang.md) | 0.5.9 | [sgl-project/sglang](https://github.com/sgl-project/sglang) | ✅ | ✅ | ~3 hr | sgl-kernel + sglang |
| [FlashAttention](flash-attention.md) | 2.8.3 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | ✅ | ✅ | ~85 min | FA2 + FA3 kernels |
| [FlashInfer](flashinfer.md) | 0.6.4 | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | ✅ | ✅ | ~3 sec | JIT (pure Python) |
| [torchao](torchao.md) | 0.16.0 | [pytorch/ao](https://github.com/pytorch/ao) | ✅ | ✅ | ~2 min | Quantization |
| [bitsandbytes](bitsandbytes.md) | 0.50.0.dev0 | [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | ✅ | ✅ | ~3 min | NF4/INT8 |
| [SageAttention](sageattention.md) | 2.2.0 | [vistralis/SageAttention](https://github.com/vistralis/SageAttention) | ✅ | ✅ | ~5 min | SM 121 fork |
| [ONNX Runtime](onnxruntime.md) | 1.25.0 | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | ✅ | ✅ | ~45 min | CUDA + TRT EP |
| [comfy-kitchen](comfy-kitchen.md) | 0.2.7 | [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) | ✅ | ✅ | ~20 sec | ComfyUI CUDA kernels |

¹ vLLM torch 2.11 wheel built from [vistralis/vllm](https://github.com/vistralis/vllm) fork with `dsv3_fused_a_gemm` SM 12.x guard.

## Candidate Packages

Under evaluation for DGX Spark builds. Not yet published.

| Package | Repo | Status | Blocker |
|---------|------|--------|---------|
| [xformers](xformers.md) | [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | 🔍 Blocked | SM 121 compile errors |
| [DeepSpeed](deepspeed.md) | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | 🔍 Evaluating | Training-focused, needs testing |
| [Triton](triton.md) | [triton-lang/triton](https://github.com/triton-lang/triton) | 🔍 Evaluating | PyTorch bundles one; custom may help SM 121 |
| [llama.cpp](llama-cpp.md) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 🔍 Evaluating | GGUF inference, different ecosystem |

## Known Platform Constraints

1. **aarch64** — most PyPI wheels are x86_64-only; CUDA extensions need source builds
2. **SM 121** — `torch.compile` crashes on PyTorch < 2.11 (`KernelMetadata` bug)
3. **CUDA 13.0** — very new; most libraries lack prebuilt wheels
4. **ABI lock-in** — wheels are tied to their base image's libtorch (NGC ≠ pip)
5. **Memory** — sgl-kernel builds OOM at MAX_JOBS > 2 on 128 GB system
