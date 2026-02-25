# ── vLLM Inference Server for DGX Spark ───────────────────────────
# High-performance LLM serving with OpenAI-compatible API.
# Pre-built aarch64 wheels from GitHub releases.
#
# Build:
#   docker build -t vistralis/vllm:torch2.11-cu130 \
#       dockerfiles/images/vllm/
#
# Run:
#   docker run --gpus all -p 8000:8000 vistralis/vllm:torch2.11-cu130 \
#       --model Qwen/Qwen3.5-35B-A3B --dtype auto
#
# ──────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=cuda13.0-torch2.11rc1-ubuntu24.04
FROM ${BASE_IMAGE}

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libnuma-dev libucx-dev curl \
    && rm -rf /var/lib/apt/lists/*

# ── Pre-built ABI-matched wheels ─────────────────────────────────
ARG WHEEL_TAG=torch2.11rc1-cu130
ARG WHEEL_BASE=https://github.com/vistralis/dgx-spark-builder/releases/download/${WHEEL_TAG}
RUN uv pip install \
    ${WHEEL_BASE}/vllm-0.16.0+cu130-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/flashinfer_python-0.6.4-py3-none-any.whl \
    ${WHEEL_BASE}/torchao-0.16.0+git6ad7c4046-cp310-abi3-linux_aarch64.whl \
    ${WHEEL_BASE}/bitsandbytes-0.50.0.dev0-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/sageattention-2.2.0-cp312-cp312-linux_aarch64.whl

# ── Additional Python deps ───────────────────────────────────────
RUN uv pip install transformers accelerate huggingface_hub hf_transfer

# ── Inference environment tuning ─────────────────────────────────
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8000

# No default model — user must pass --model at runtime
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server", "--host", "0.0.0.0", "--port", "8000"]
