#!/usr/bin/env bash
# Probe NGC images for installed packages
# Usage: ./probe_images.sh
# Output: /tmp/ngc_probe_results.txt

set -euo pipefail

RESULTS="/tmp/ngc_probe_results.txt"
> "$RESULTS"

PROBE_SCRIPT='
import sys, json, subprocess
info = {"python": sys.version.split()[0]}
pkgs = {
    "torch": "torch", "tensorrt": "tensorrt", "onnxruntime": "onnxruntime",
    "onnx": "onnx", "torch_tensorrt": "torch_tensorrt", "vllm": "vllm",
    "transformers": "transformers", "diffusers": "diffusers",
    "triton": "triton", "xformers": "xformers", "torchao": "torchao",
    "bitsandbytes": "bitsandbytes", "modelopt": "modelopt",
    "accelerate": "accelerate", "optimum": "optimum",
}
for name, mod in pkgs.items():
    try:
        m = __import__(mod)
        info[name] = getattr(m, "__version__", "installed")
    except:
        info[name] = None
# CUDA version
try:
    import torch
    info["cuda"] = torch.version.cuda or "N/A"
except:
    try:
        r = subprocess.run(["nvcc","--version"], capture_output=True, text=True)
        for l in r.stdout.split("\n"):
            if "release" in l: info["cuda"] = l.split("release ")[-1].split(",")[0]
    except:
        info["cuda"] = None
# ORT providers (without GPU)
try:
    import onnxruntime as ort
    info["ort_providers"] = ort.get_available_providers()
except:
    info["ort_providers"] = None
# Image size
try:
    r = subprocess.run(["du","-sh","/"], capture_output=True, text=True, timeout=5)
    info["image_size"] = r.stdout.split()[0] if r.stdout else "?"
except:
    info["image_size"] = "?"
print(json.dumps(info))
'

IMAGES=(
    "nvcr.io/nvidia/pytorch:25.11-py3"
    "nvcr.io/nvidia/pytorch:26.01-py3"
    "nvcr.io/nvidia/tensorrt:25.11-py3"
    "nvcr.io/nvidia/tensorrt:26.01-py3"
    "nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04"
    "nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-runtime-ubuntu24.04"
    "nvcr.io/nvidia/cuda-dl-base:26.01-cuda13.1-inference-devel-ubuntu24.04"
    "nvcr.io/nvidia/cuda-dl-base:26.01-cuda13.1-inference-runtime-ubuntu24.04"
    "nvcr.io/nvidia/vllm:25.11-py3"
    "nvcr.io/nvidia/vllm:26.01-py3"
)

echo "Probing ${#IMAGES[@]} NGC images..." | tee -a "$RESULTS"
echo "" >> "$RESULTS"

for img in "${IMAGES[@]}"; do
    echo "─── Probing: $img" | tee -a "$RESULTS"
    # Timeout after 60s per image
    result=$(timeout 60 docker run --rm "$img" python3 -c "$PROBE_SCRIPT" 2>/dev/null || echo '{"error":"probe failed or timed out"}')
    echo "  $result" >> "$RESULTS"
    echo "  $result"
    echo "" >> "$RESULTS"
done

echo "Done! Results in $RESULTS" | tee -a "$RESULTS"
