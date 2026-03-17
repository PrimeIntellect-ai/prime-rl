#!/usr/bin/env bash
# Install DeepEP kernels for expert-parallel disaggregated inference.
#
# Builds NVSHMEM + DeepEP from source and installs into the project venv.
# Required for disaggregated prefill/decode deployments with expert parallelism.
#
# Auto-detects the CUDA toolkit version that matches the installed PyTorch
# and looks for it at /usr/local/cuda-X.Y. Fails if not found.
#
# Prerequisites:
#   - Matching CUDA toolkit installed (e.g. `sudo apt install cuda-toolkit-12-8`)
#   - For multi-node: IBGDA driver configured (see --configure-drivers)
#
# Usage:
#   bash scripts/install_ep_kernels.sh
#
# Options:
#   --workspace DIR       Build directory (default: ./ep_kernels_workspace)
#   --deepep-ref REF      DeepEP commit hash (default: 73b6ea4)
#   --nvshmem-ver VER     NVSHMEM version (default: 3.3.24)
#   --configure-drivers   Also configure IBGDA drivers (requires sudo, needs reboot)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

DEEPEP_COMMIT_HASH="73b6ea4"
NVSHMEM_VER="3.3.24"
WORKSPACE="$REPO_ROOT/ep_kernels_workspace"
CONFIGURE_DRIVERS=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --workspace)         WORKSPACE="$2";          shift 2 ;;
        --deepep-ref)        DEEPEP_COMMIT_HASH="$2"; shift 2 ;;
        --nvshmem-ver)       NVSHMEM_VER="$2";        shift 2 ;;
        --configure-drivers) CONFIGURE_DRIVERS=1;     shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Skip if DeepEP is already installed
if python -c "import deep_ep" 2>/dev/null; then
    echo "DeepEP already installed, skipping."
    exit 0
fi

# ── Auto-detect CUDA toolkit matching torch ───────────────────────────────────
TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
CUDA_MAJOR_MINOR=$(echo "$TORCH_CUDA_VER" | grep -oP '^\d+\.\d+')
CUDA_MAJOR=$(echo "$CUDA_MAJOR_MINOR" | cut -d. -f1)

CUDA_HOME="/usr/local/cuda-${CUDA_MAJOR_MINOR}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: Could not find CUDA toolkit matching torch (cuda ${TORCH_CUDA_VER}) at ${CUDA_HOME}" >&2
    echo "Install it with: sudo apt install cuda-toolkit-${CUDA_MAJOR_MINOR//./-}" >&2
    exit 1
fi
export CUDA_HOME

# Verify versions actually match
NVCC_VER=$("$CUDA_HOME/bin/nvcc" --version | grep -oP 'release \K[\d.]+')
echo "Torch CUDA: ${TORCH_CUDA_VER}, nvcc: ${NVCC_VER} (${CUDA_HOME})"

# ── Auto-detect GPU architecture ──────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits -i 0 2>/dev/null)
case "$GPU_NAME" in
    *H100*|*H200*|*H800*)  TORCH_CUDA_ARCH_LIST="9.0" ;;
    *B100*|*B200*|*GB200*)  TORCH_CUDA_ARCH_LIST="10.0" ;;
    *)
        echo "Could not auto-detect GPU arch from '$GPU_NAME'. Set TORCH_CUDA_ARCH_LIST manually." >&2
        echo "  Hopper (H100/H200): TORCH_CUDA_ARCH_LIST=9.0" >&2
        echo "  Blackwell (B200):   TORCH_CUDA_ARCH_LIST=10.0" >&2
        exit 1
        ;;
esac
export TORCH_CUDA_ARCH_LIST

echo "================================================================"
echo " Installing DeepEP kernels"
echo " CUDA:       ${CUDA_HOME} (${NVCC_VER})"
echo " GPU:        ${GPU_NAME} (arch ${TORCH_CUDA_ARCH_LIST})"
echo " DeepEP:     ${DEEPEP_COMMIT_HASH}"
echo " NVSHMEM:    ${NVSHMEM_VER}"
echo " Workspace:  ${WORKSPACE}"
echo "================================================================"

mkdir -p "$WORKSPACE"

# ── Step 1: Install build deps ────────────────────────────────────────────────
echo ""
echo "--- Installing build dependencies ---"
cd "$REPO_ROOT"
uv pip install cmake ninja

# ── Step 2: Download and extract NVSHMEM ──────────────────────────────────────
echo ""
echo "--- Setting up NVSHMEM ${NVSHMEM_VER} ---"

ARCH=$(uname -m)
case "${ARCH,,}" in
    x86_64|amd64)   NVSHMEM_SUBDIR="linux-x86_64" ;;
    aarch64|arm64)   NVSHMEM_SUBDIR="linux-sbsa" ;;
    *) echo "Unsupported architecture: ${ARCH}" >&2; exit 1 ;;
esac

NVSHMEM_DIR="$WORKSPACE/nvshmem"
if [ ! -d "$NVSHMEM_DIR/lib" ]; then
    NVSHMEM_FILE="libnvshmem-${NVSHMEM_SUBDIR}-${NVSHMEM_VER}_cuda${CUDA_MAJOR}-archive.tar.xz"
    NVSHMEM_URL="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/${NVSHMEM_SUBDIR}/${NVSHMEM_FILE}"

    echo "Downloading ${NVSHMEM_URL}"
    curl -fSL "${NVSHMEM_URL}" -o "$WORKSPACE/${NVSHMEM_FILE}"
    tar -xf "$WORKSPACE/${NVSHMEM_FILE}" -C "$WORKSPACE"
    mv "$WORKSPACE/${NVSHMEM_FILE%.tar.xz}" "$NVSHMEM_DIR"
    rm -f "$WORKSPACE/${NVSHMEM_FILE}"
    rm -rf "$NVSHMEM_DIR/lib/bin" "$NVSHMEM_DIR/lib/share"
    echo "NVSHMEM extracted to ${NVSHMEM_DIR}"
else
    echo "NVSHMEM already present at ${NVSHMEM_DIR}, skipping download"
fi

export CMAKE_PREFIX_PATH="${NVSHMEM_DIR}/lib/cmake:${CMAKE_PREFIX_PATH:-}"
export NVSHMEM_DIR

# ── Step 3: Build and install DeepEP ──────────────────────────────────────────
echo ""
echo "--- Building DeepEP (${DEEPEP_COMMIT_HASH}) ---"

DEEPEP_DIR="$WORKSPACE/DeepEP"
if [ ! -d "$DEEPEP_DIR/.git" ]; then
    git clone https://github.com/deepseek-ai/DeepEP "$DEEPEP_DIR"
fi

cd "$DEEPEP_DIR"
git fetch origin
git checkout "$DEEPEP_COMMIT_HASH"

WHEEL_DIR="$REPO_ROOT/deps"
mkdir -p "$WHEEL_DIR"
python setup.py bdist_wheel --dist-dir "$WHEEL_DIR"

WHEEL=$(ls "$WHEEL_DIR"/deep_ep*.whl | head -1)
echo ""
echo "--- DeepEP wheel built at: $WHEEL ---"

# ── Step 4 (optional): Configure IBGDA drivers ───────────────────────────────
if [ "$CONFIGURE_DRIVERS" -eq 1 ]; then
    echo ""
    echo "--- Configuring IBGDA drivers (requires sudo) ---"
    echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | sudo tee -a /etc/modprobe.d/nvidia.conf
    if command -v update-initramfs &> /dev/null; then
        sudo update-initramfs -u
    elif command -v dracut &> /dev/null; then
        sudo dracut --force
    else
        echo "No supported initramfs update tool found." >&2
        exit 1
    fi
    echo ""
    echo "IBGDA configured. Please REBOOT the system to apply changes."
fi

echo ""
echo "================================================================"
echo " DeepEP installation complete"
echo " NVSHMEM: ${NVSHMEM_DIR}"
echo " To verify: uv run python -c 'import deep_ep; print(deep_ep.__file__)'"
echo "================================================================"
