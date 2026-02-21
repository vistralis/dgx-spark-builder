# ComfyUI + Qwen3-TTS on DGX Spark (Blackwell, SM 121)
#
# Layered on top of the published ComfyUI image, adds the
# flybirdxx/ComfyUI-Qwen-TTS custom node and its dependencies.
#
# Usage:
#   docker build -f dockerfiles/comfyui-qwen-tts/Dockerfile \
#       -t comfyui-spark:qwen-tts .
#
ARG BASE_IMAGE=vistralis/comfyui:cuda13.0-torch2.10-ubuntu24.04
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

# ── Custom nodes (pinned to known-good commits) ──────────────────
# Qwen3-TTS (flybirdxx) — voice synthesis & cloning
ARG QWEN_TTS_COMMIT=17207f8b9d1e2e92cb5cf0e487222656757bedf5
RUN git clone https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git \
    /opt/comfyui/custom_nodes/ComfyUI-Qwen-TTS \
    && cd /opt/comfyui/custom_nodes/ComfyUI-Qwen-TTS \
    && git checkout ${QWEN_TTS_COMMIT}

# Custom-Scripts (pythongosssss) — ShowText, MarkdownNote, etc.
ARG CUSTOM_SCRIPTS_COMMIT=609f3afaa74b2f88ef9ce8d939626065e3247469
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git \
    /opt/comfyui/custom_nodes/ComfyUI-Custom-Scripts \
    && cd /opt/comfyui/custom_nodes/ComfyUI-Custom-Scripts \
    && git checkout ${CUSTOM_SCRIPTS_COMMIT}

# LayerStyle (chflame163) — PurgeVRAM, layer utilities
ARG LAYER_STYLE_COMMIT=d94bef1ee5ed3656f5ff1bb2830a4ffd94f40935
RUN git clone https://github.com/chflame163/ComfyUI_LayerStyle.git \
    /opt/comfyui/custom_nodes/ComfyUI_LayerStyle \
    && cd /opt/comfyui/custom_nodes/ComfyUI_LayerStyle \
    && git checkout ${LAYER_STYLE_COMMIT} \
    && uv pip install -r requirements.txt

# ── Runtime config ────────────────────────────────────────────────
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
