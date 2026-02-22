# ComfyUI + Qwen3-TTS on DGX Spark

Voice synthesis, cloning, and fine-tuning using [DarioFT/ComfyUI-Qwen3-TTS](https://github.com/DarioFT/ComfyUI-Qwen3-TTS) on DGX Spark (Blackwell, SM 121).

## Quick Start

```bash
# Build
git clone https://github.com/vistralis/dgx-spark-builder.git
cd dgx-spark-builder
docker build -f dockerfiles/comfyui-qwen-tts/Dockerfile \
    -t comfyui-spark:qwen-tts .

# Create persistent storage
mkdir -p ~/comfyui/{models,output,input}

# Run (open http://localhost:8188/)
# Models download automatically on first use (~14 GB)
docker run --gpus all --ipc=host -p 8188:8188 \
    -v ~/comfyui/models:/opt/comfyui/models \
    -v ~/comfyui/output:/opt/comfyui/output \
    -v ~/comfyui/input:/opt/comfyui/input \
    comfyui-spark:qwen-tts
```

Models are stored under `~/comfyui/models/Qwen3-TTS/` and persist across container restarts.

## Available Nodes

| Node | Description |
|---|---|
| **Qwen3 TTS Generate** | Text-to-speech with voice design (gender, pitch, speed) |
| **Qwen3 TTS Clone** | Voice cloning from a reference audio sample |
| **Qwen3 TTS DatasetMaker** | Prepare `.wav`/`.txt` pairs for fine-tuning |
| **Qwen3 TTS DataPrep** | Tokenize audio into codec codes for training |
| **Qwen3 TTS FineTune** | Train a custom voice with 8-bit AdamW |
| **Qwen3 TTS CustomVoice** | Inference using a fine-tuned checkpoint |

Example workflows are pre-loaded under **Workflows → default**.

## Fine-Tuning a Custom Voice

### 1. Prepare Training Data

You need **10–50 short audio clips** (5–15 seconds each) of a single speaker, plus matching text transcriptions.

**Directory structure:**
```
my-dataset/
├── clip_001.wav
├── clip_001.txt     # transcription of clip_001.wav
├── clip_002.wav
├── clip_002.txt
├── ...
└── ref.wav          # reference clip (excluded from training)
```

**Audio requirements:**
- Format: `.wav` (16-bit PCM, mono recommended)
- Sample rate: any (resampled automatically to 24 kHz)
- Clean speech, minimal background noise
- Consistent recording conditions across clips

**Generating transcriptions with Whisper:**

If you don't have `.txt` files, use `faster-whisper` (pre-installed) to generate them:

```bash
docker run --gpus all --rm -it \
    -v ~/my-dataset:/data \
    comfyui-spark:qwen-tts \
    python -c "
from faster_whisper import WhisperModel
import os, glob

model = WhisperModel('large-v3', device='cuda', compute_type='float16')
for wav in sorted(glob.glob('/data/*.wav')):
    segments, _ = model.transcribe(wav, language='en')
    text = ' '.join(s.text.strip() for s in segments)
    txt_path = wav.rsplit('.', 1)[0] + '.txt'
    with open(txt_path, 'w') as f:
        f.write(text)
    print(f'{os.path.basename(wav)}: {text}')
"
```

**Converting audio with ffmpeg** (included in the image):

```bash
# Convert mp3 to wav
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav

# Trim a clip (start at 1:05, duration 10s)
ffmpeg -i long_recording.wav -ss 1:05 -t 10 -ar 24000 -ac 1 clip_001.wav
```

### 2. Mount Data and Run

```bash
docker run --gpus all --ipc=host -p 8188:8188 \
    -v ~/comfyui/models:/opt/comfyui/models \
    -v ~/comfyui/output:/opt/comfyui/output \
    -v ~/comfyui/input:/opt/comfyui/input \
    -v ~/my-dataset:/opt/comfyui/models/qwen-tts/datasets/my-speaker \
    comfyui-spark:qwen-tts
```

### 3. Run the Workflow

In the ComfyUI browser UI:

1. **DatasetMaker** — point `dataset_path` to `/opt/comfyui/models/qwen-tts/datasets/my-speaker`
2. **DataPrep** — tokenizes audio into codec codes
3. **FineTune** — trains the model (default: 10 epochs, lr=2e-6, 8-bit AdamW)
4. **CustomVoice** — inference with your trained checkpoint

### Training Tips

| Parameter | Default | Notes |
|---|---|---|
| Learning rate | 2e-6 | Lower (1e-6) for <20 clips, higher (5e-6) for 50+ |
| Epochs | 10 | Monitor loss — stop if it plateaus |
| Batch size | 2 | Increase to 4–8 if VRAM allows |
| Grad accumulation | 4 | Effective batch = batch_size × grad_accum |
| Sub-talker weight | 0.3 | Balances main vs sub-talker loss |

## Migrating from Previous Models

If you have models from a previous install at `~/comfyui/models/qwen-tts/`:

```bash
mkdir -p ~/comfyui/models/Qwen3-TTS
cd ~/comfyui/models/Qwen3-TTS
ln -sf ../qwen-tts/Qwen3-TTS-12Hz-1.7B-Base .
ln -sf ../qwen-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign .
ln -sf ../qwen-tts/Qwen3-TTS-Tokenizer-12Hz .
```

## Included Tools

| Tool | Version | Purpose |
|---|---|---|
| [qwen-tts](https://github.com/QwenLM/Qwen3-TTS) | 0.1.1 (from source) | Core TTS engine |
| [DarioFT/ComfyUI-Qwen3-TTS](https://github.com/DarioFT/ComfyUI-Qwen3-TTS) | 17c22ad | ComfyUI nodes |
| [VHS](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | latest | Audio preview |
| faster-whisper | latest | ASR / transcription |
| bitsandbytes | 0.50.0.dev0 | 8-bit optimizer (CUDA 13.0) |
| ffmpeg | system | Audio format conversion |
