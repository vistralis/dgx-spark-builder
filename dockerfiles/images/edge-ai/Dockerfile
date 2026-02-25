# ── Edge AI Development Image for DGX Spark ──────────────────────
# Everything-included image for inference, fine-tuning, and
# experimentation. All pre-built aarch64 wheels installed.
#
# Build:
#   docker build -t vistralis/edge-ai:torch2.11-cu130 \
#       dockerfiles/images/edge-ai/
#
# Run (interactive):
#   docker run --gpus all -it -p 8888:8888 \
#       vistralis/edge-ai:torch2.11-cu130
#
# ──────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=cuda13.0-torch2.11rc1-ubuntu24.04
FROM ${BASE_IMAGE}

# ── System deps ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libnuma-dev libucx-dev curl wget \
    libgl1 libglib2.0-0 libsndfile1 sox \
    libglfw3-dev libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# ── All pre-built ABI-matched wheels ─────────────────────────────
# sglang pins flashinfer-python==0.6.3 but we have 0.6.4; use --no-deps
ARG WHEEL_TAG=torch2.11rc1-cu130
ARG WHEEL_BASE=https://github.com/vistralis/dgx-spark-builder/releases/download/${WHEEL_TAG}
RUN uv pip install --no-deps \
    ${WHEEL_BASE}/sglang-0.5.9-py3-none-any.whl \
    && uv pip install \
    ${WHEEL_BASE}/vllm-0.16.0+cu130-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/sgl_kernel-0.3.21-cp310-abi3-linux_aarch64.whl \
    ${WHEEL_BASE}/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/flashinfer_python-0.6.4-py3-none-any.whl \
    ${WHEEL_BASE}/torchao-0.16.0+git6ad7c4046-cp310-abi3-linux_aarch64.whl \
    ${WHEEL_BASE}/bitsandbytes-0.50.0.dev0-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/sageattention-2.2.0-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/onnxruntime_gpu-1.25.0-cp312-cp312-linux_aarch64.whl \
    ${WHEEL_BASE}/comfy_kitchen-0.2.7-cp312-abi3-linux_aarch64.whl

# ── ML / serving stack ───────────────────────────────────────────
RUN uv pip install \
    transformers accelerate peft \
    diffusers safetensors \
    datasets huggingface_hub hf_transfer \
    gradio jupyter \
    deepspeed \
    opencv-python-headless pillow \
    sentencepiece protobuf einops

# ── Environment ──────────────────────────────────────────────────
ENV HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 8000 8888 30000

# Interactive shell by default
CMD ["/bin/bash"]
