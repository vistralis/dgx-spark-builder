# ComfyUI + Qwen3-TTS on DGX Spark (Blackwell, SM 121)
#
# Layered on top of the base ComfyUI image, adds the
# flybirdxx/ComfyUI-Qwen-TTS custom node and its dependencies.
#
# Usage:
#   # 1. Build base comfyui image first
#   docker build -f dockerfiles/comfyui/Dockerfile \
#       --build-context wheels=wheels/pip-torch2.10-cu130 \
#       -t comfyui-spark .
#
#   # 2. Build qwen-tts specialization
#   docker build -f dockerfiles/comfyui-qwen-tts/Dockerfile \
#       -t comfyui-spark:qwen-tts .
#
ARG BASE_IMAGE=comfyui-spark
FROM ${BASE_IMAGE}

# ── Qwen-TTS Python dependencies ─────────────────────────────────
# Pin transformers==4.57.3 — v5.x has breaking config changes that
# cause 'pad_token_id' AttributeError. See:
# https://github.com/flybirdxx/ComfyUI-Qwen-TTS/issues/74
RUN uv pip install \
    torchaudio --index-url https://download.pytorch.org/whl/cu130 \
    && uv pip install \
    "transformers==4.57.3" \
    librosa \
    soundfile \
    accelerate \
    einops \
    tiktoken \
    sentencepiece \
    sox \
    huggingface_hub \
    safetensors \
    "scipy>=1.11.0"

# ── Install Qwen-TTS custom node ─────────────────────────────────
ARG QWEN_TTS_REPO=https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git
ARG QWEN_TTS_VERSION=main
RUN git clone --depth=1 --branch ${QWEN_TTS_VERSION} \
    ${QWEN_TTS_REPO} /opt/comfyui/custom_nodes/ComfyUI-Qwen-TTS

# ── Runtime config ────────────────────────────────────────────────
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
