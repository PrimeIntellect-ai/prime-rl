#!/usr/bin/env bash
# Build and install DeepGEMM from source for FP8 MoE inference.
#
# Uses the commit recommended by vLLM for compatibility with vLLM 0.17+.
# Requires CUDA 12.8+ and a Hopper/Blackwell GPU.
#
# Usage:
#   bash scripts/install_deep_gemm.sh
#
# Options:
#   --ref REF             DeepGEMM commit hash (default: 477618c)
#   --wheel-dir DIR       Output wheel to DIR instead of installing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DEEPGEMM_GIT_REPO="https://github.com/deepseek-ai/DeepGEMM.git"
DEEPGEMM_GIT_REF="477618cd51baffca09c4b0b87e97c03fe827ef03"
WHEEL_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ref)       DEEPGEMM_GIT_REF="$2"; shift 2 ;;
        --wheel-dir) WHEEL_DIR="$2";        shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Skip if already installed at the right version
INSTALLED_VER=$(python -c "import deep_gemm; print(deep_gemm.__version__)" 2>/dev/null || echo "")
SHORT_REF="${DEEPGEMM_GIT_REF:0:7}"
if [[ "$INSTALLED_VER" == *"$SHORT_REF"* ]]; then
    echo "DeepGEMM ${INSTALLED_VER} already installed, skipping."
    exit 0
fi

# Auto-detect CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "")
if [ -z "$CUDA_VERSION" ]; then
    echo "ERROR: nvcc not found. CUDA toolkit required." >&2
    exit 1
fi

CUDA_MAJOR="${CUDA_VERSION%%.*}"
CUDA_MINOR="${CUDA_VERSION#*.}"
CUDA_MINOR="${CUDA_MINOR%%.*}"
if [ "$CUDA_MAJOR" -lt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 8 ]; }; then
    echo "Skipping DeepGEMM (requires CUDA 12.8+, got ${CUDA_VERSION})"
    exit 0
fi

echo "================================================================"
echo " Building DeepGEMM (${SHORT_REF})"
echo " CUDA: ${CUDA_VERSION}"
echo "================================================================"

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

git clone --recurse-submodules "$DEEPGEMM_GIT_REPO" "$TMPDIR/DeepGEMM" 2>&1
cd "$TMPDIR/DeepGEMM"
git checkout "$DEEPGEMM_GIT_REF"
git submodule update --init --recursive

cd "$REPO_ROOT"

if [ -n "$WHEEL_DIR" ]; then
    mkdir -p "$WHEEL_DIR"
    uv build --no-build-isolation --wheel --out-dir "$WHEEL_DIR" "$TMPDIR/DeepGEMM"
    echo ""
    echo "Wheel built:"
    ls -lh "$WHEEL_DIR"/deep_gemm*.whl
else
    uv pip install --no-build-isolation --reinstall "$TMPDIR/DeepGEMM"
    echo ""
    echo "Installed: $(python -c 'import deep_gemm; print(deep_gemm.__version__)')"
fi

echo "================================================================"
echo " DeepGEMM build complete"
echo "================================================================"
