# ComfyUI + Qwen3-TTS on DGX Spark (Blackwell, SM 121)
#
# Uses DarioFT/ComfyUI-Qwen3-TTS extension with qwen-tts installed
# from source (QwenLM/Qwen3-TTS repo) so we can evaluate upstream PRs.
#
# Usage:
#   docker build -f dockerfiles/comfyui-qwen-tts/Dockerfile \
#       -t comfyui-spark:qwen-tts .
#
#   docker run --gpus all --ipc=host -p 8188:8188 \
#       -v ~/comfyui/models:/opt/comfyui/models \
#       -v ~/comfyui/output:/opt/comfyui/output \
#       -v ~/comfyui/input:/opt/comfyui/input \
#       comfyui-spark:qwen-tts
#
ARG BASE_IMAGE=vistralis/comfyui:cuda13.0-torch2.10-ubuntu24.04
FROM ${BASE_IMAGE}

# ── Python dependencies ───────────────────────────────────────────
# Install BEFORE qwen-tts so that the base image's custom
# bitsandbytes (CUDA 13.0) is not overwritten by PyPI.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN uv pip install \
    torchaudio --index-url https://download.pytorch.org/whl/cu130 \
    && uv pip install \
    accelerate \
    "transformers==4.57.3" \
    librosa \
    soundfile \
    einops \
    tiktoken \
    sentencepiece \
    sox \
    onnxruntime \
    tensorboard \
    huggingface_hub \
    safetensors \
    "scipy>=1.11.0" \
    faster-whisper

# ── Install qwen-tts from source (not PyPI) ──────────────────────
# Clone the official repo so we can cherry-pick upstream PRs.
# Use --no-deps to avoid re-installing deps from PyPI which would
# overwrite our custom bitsandbytes build for CUDA 13.0.
ARG QWEN_TTS_COMMIT=6cafe5582caea83df269c36b1ce62d953a9cc66b
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /tmp/Qwen3-TTS \
    && cd /tmp/Qwen3-TTS \
    && git checkout ${QWEN_TTS_COMMIT} \
    && uv pip install --no-deps -e . \
    && rm -rf /tmp/Qwen3-TTS/.git

# ── ComfyUI custom nodes ─────────────────────────────────────────
# DarioFT/ComfyUI-Qwen3-TTS — voice synthesis, cloning & fine-tuning
ARG DARIOFT_COMMIT=17c22adb80a63c1c51bf74549e71a5cf218f4e1b
RUN git clone https://github.com/DarioFT/ComfyUI-Qwen3-TTS.git \
    /opt/comfyui/custom_nodes/ComfyUI-Qwen3-TTS \
    && cd /opt/comfyui/custom_nodes/ComfyUI-Qwen3-TTS \
    && git checkout ${DARIOFT_COMMIT} \
    && mkdir -p /opt/comfyui/user/default/workflows \
    && cp example_workflows/*.json /opt/comfyui/user/default/workflows/

# VHS (Video Helper Suite) — audio preview in workflows
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    /opt/comfyui/custom_nodes/ComfyUI-VideoHelperSuite \
    && cd /opt/comfyui/custom_nodes/ComfyUI-VideoHelperSuite \
    && uv pip install -r requirements.txt

# ── Runtime config ────────────────────────────────────────────────
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
