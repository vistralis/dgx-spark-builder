#!/usr/bin/env bash
# Build custom CUDA-compiled wheels for DGX SPARK GB10 (Blackwell, SM 121)
# Wheels exported via docker build --output, organized by torch+cuda version.
#
# Usage:
#   ./build_wheels.sh                           # build all with defaults
#   ./build_wheels.sh onnxruntime               # build one
#   ./build_wheels.sh onnxruntime bitsandbytes  # build multiple
#
# Base image (determines output directory):
#   BASE_IMAGE=cuda13.0-torch2.10-ubuntu24.04 ./build_wheels.sh   # → wheels/torch2.10-cu130/
#   BASE_IMAGE=cuda13.0-torch2.11rc1-ubuntu24.04 ./build_wheels.sh # → wheels/torch2.11rc1-cu130/
#
# Version pinning:
#   ORT_VERSION=v1.21.0 ./build_wheels.sh onnxruntime
#   SAGE_VERSION=blackwell-sm121 ./build_wheels.sh sageattention
#   FLASH_ATTN_VERSION=v2.8.3 ./build_wheels.sh flash-attention
#   BNB_VERSION=0.45.3 ./build_wheels.sh bitsandbytes
#   XFORMERS_VERSION=v0.0.30 ./build_wheels.sh xformers
#   TORCHAO_VERSION=v0.14.0 ./build_wheels.sh torchao
#   VLLM_VERSION=v0.16.0 ./build_wheels.sh vllm
#
# Output layout (matches GitHub release tags):
#   wheels/
#   ├── torch2.10-cu130/            # PyTorch 2.10 + CUDA 13.0
#   └── torch2.11rc1-cu130/         # PyTorch 2.11 RC1 + CUDA 13.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
WHEELS_DIR="${PROJECT_DIR}/wheels"
DOCKERFILES_DIR="${PROJECT_DIR}/dockerfiles"

# Parse arguments (all positional — target names)
set -- "$@"

# Configurable base image
BASE_IMAGE="${BASE_IMAGE:-cuda13.0-torch2.10-ubuntu24.04}"

# Derive output directory from base image name
# cuda13.0-torch2.10-ubuntu24.04 → torch2.10-cu130
# cuda13.0-torch2.11rc1-ubuntu24.04 → torch2.11rc1-cu130
derive_output_dir() {
    local img="$1"
    # Strip registry prefix if present
    [[ "$img" == *":"* ]] && img="${img##*:}"
    # Try to extract torch version and cuda version
    local torch_ver cuda_ver
    torch_ver=$(echo "$img" | grep -oP 'torch\K[0-9a-z.]+' || echo "")
    cuda_ver=$(echo "$img" | grep -oP 'cuda\K[0-9.]+' || echo "")
    if [[ -n "$torch_ver" && -n "$cuda_ver" ]]; then
        echo "torch${torch_ver}-cu${cuda_ver//.}"
    else
        echo "$img"
    fi
}

OUTPUT_DIR=$(derive_output_dir "$BASE_IMAGE")

# Default versions (override via env vars)
ORT_VERSION="${ORT_VERSION:-main}"
ORT_REPO="${ORT_REPO:-microsoft/onnxruntime}"
XFORMERS_VERSION="${XFORMERS_VERSION:-main}"
TORCHAO_VERSION="${TORCHAO_VERSION:-main}"
BNB_VERSION="${BNB_VERSION:-main}"
SAGE_VERSION="${SAGE_VERSION:-blackwell-sm121}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-v2.8.3}"
VLLM_VERSION="${VLLM_VERSION:-v0.16.0}"
COMFY_KITCHEN_VERSION="${COMFY_KITCHEN_VERSION:-v0.2.7}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-v0.6.4}"

# ── Detect CUDA version from the base image ──────────────────────────
detect_cuda_version() {
    local cuda_ver
    cuda_ver=$(docker run --rm "${BASE_IMAGE}" \
        nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "")
    if [ -z "$cuda_ver" ]; then
        echo "unknown"
    else
        echo "cu$(echo "$cuda_ver" | tr -d '.')"
    fi
}

# Output directory
IMAGE_DIR="${WHEELS_DIR}/${OUTPUT_DIR}"

# Dynamic-width box based on content
_hdr_lines=(
    "Base image: ${BASE_IMAGE}"
    "Output dir: ${OUTPUT_DIR}"
    "Output:     ${IMAGE_DIR}/"
)

_title="DGX Spark Builder — Wheel Build"
_maxlen=${#_title}
for _l in "${_hdr_lines[@]}"; do
    (( ${#_l} > _maxlen )) && _maxlen=${#_l}
done
_repeat() { python3 -c "print('$1' * $2, end='')"; }
_w=$(( _maxlen + 4 ))
_bar=$(_repeat '═' "$_w")
printf '╔%s╗\n' "$_bar"
printf '║  %-*s  ║\n' "$((_w - 4))" "$_title"
printf '╠%s╣\n' "$_bar"
for _l in "${_hdr_lines[@]}"; do
    printf '║  %-*s  ║\n' "$((_w - 4))" "$_l"
done
printf '╚%s╝\n' "$_bar"

echo ""
echo "  Detecting CUDA version from base image..."
CUDA_TAG=$(detect_cuda_version)
echo "  CUDA tag: ${CUDA_TAG}"

mkdir -p "${IMAGE_DIR}"

echo ""

# ── Build a single wheel ─────────────────────────────────────────────
build_wheel() {
    local name="$1"
    local version_arg="$2"
    local version_val="$3"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Building ${name} @ ${version_val}"
    echo "  Base: ${BASE_IMAGE} → ${OUTPUT_DIR}/"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    local -a extra_args=()
    if [[ "$name" == "onnxruntime" && "${ORT_REPO}" != "microsoft/onnxruntime" ]]; then
        extra_args+=(--build-arg "ORT_REPO=${ORT_REPO}")
    fi

    docker build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "${version_arg}=${version_val}" \
        ${extra_args[@]+"${extra_args[@]}"} \
        --output "type=local,dest=${IMAGE_DIR}" \
        -f "${DOCKERFILES_DIR}/${name}/Dockerfile" \
        "${DOCKERFILES_DIR}/${name}/"

    echo ""
    echo "  ✓ ${name} wheel → ${IMAGE_DIR}/"
}

# ── Target map ────────────────────────────────────────────────────────
declare -A WHEEL_MAP=(
    [onnxruntime]="ORT_VERSION:${ORT_VERSION}"
    [xformers]="XFORMERS_VERSION:${XFORMERS_VERSION}"
    [torchao]="TORCHAO_VERSION:${TORCHAO_VERSION}"
    [bitsandbytes]="BNB_VERSION:${BNB_VERSION}"
    [sageattention]="SAGE_VERSION:${SAGE_VERSION}"
    [flash-attention]="FLASH_ATTN_VERSION:${FLASH_ATTN_VERSION}"
    [vllm]="VLLM_VERSION:${VLLM_VERSION}"
    [comfy-kitchen]="COMFY_KITCHEN_VERSION:${COMFY_KITCHEN_VERSION}"
    [flashinfer]="FLASHINFER_VERSION:${FLASHINFER_VERSION}"
)

# Build order: lighter builds first, onnxruntime last (longest)
if [ $# -eq 0 ]; then
    targets=("bitsandbytes" "sageattention" "flash-attention" "onnxruntime")
else
    targets=("$@")
fi

echo "  Targets: ${targets[*]}"

for target in "${targets[@]}"; do
    if [[ -z "${WHEEL_MAP[$target]+x}" ]]; then
        echo "ERROR: Unknown target '${target}'. Valid: ${!WHEEL_MAP[*]}"
        exit 1
    fi
    IFS=':' read -r arg_name arg_val <<< "${WHEEL_MAP[$target]}"
    build_wheel "$target" "$arg_name" "$arg_val"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done! Base: ${BASE_IMAGE} → ${OUTPUT_DIR}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Wheels:"
ls -lh "${IMAGE_DIR}"/*.whl 2>/dev/null || echo "  (none)"
echo ""
