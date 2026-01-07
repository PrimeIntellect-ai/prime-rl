#!/bin/bash
# Setup script for transformer-engine with pip-installed NVIDIA libraries

set -e

echo "=== Transformer Engine Setup ==="
echo

# Detect site-packages path
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
NVIDIA_PATH="$SITE_PACKAGES/nvidia"

if [ ! -d "$NVIDIA_PATH" ]; then
    echo "Error: NVIDIA packages not found at $NVIDIA_PATH"
    echo "Please ensure nvidia-cudnn-cu12 and other NVIDIA packages are installed."
    exit 1
fi

echo "Found NVIDIA packages at: $NVIDIA_PATH"

# Detect CUDA version from installed packages or nvidia-smi
detect_cuda_version() {
    # Try to detect from pip packages first (e.g., nvidia-cuda-runtime-cu12)
    CUDA_PKG=$(pip list 2>/dev/null | grep -oP 'nvidia-cuda-runtime-cu\K[0-9]+' | head -1)
    if [ -n "$CUDA_PKG" ]; then
        echo "$CUDA_PKG"
        return
    fi
    
    # Fallback: try nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        # Map driver to CUDA (rough estimate)
        if nvidia-smi 2>/dev/null | grep -qE "CUDA Version: 13"; then
            echo "13"
            return
        elif nvidia-smi 2>/dev/null | grep -qE "CUDA Version: 12"; then
            echo "12"
            return
        fi
    fi
    
    # Fallback: try nvcc
    if command -v nvcc &>/dev/null; then
        NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1)
        if [ -n "$NVCC_VER" ]; then
            echo "$NVCC_VER"
            return
        fi
    fi
    
    echo ""
}

CUDA_MAJOR=$(detect_cuda_version)
if [ -z "$CUDA_MAJOR" ]; then
    echo "Warning: Could not detect CUDA version. Defaulting to cu12."
    CUDA_MAJOR="12"
fi
echo "Detected CUDA major version: $CUDA_MAJOR"

# Determine the correct extra
if [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
    TE_EXTRA="te-cu13"
else
    TE_EXTRA="te-cu12"
fi
echo "Using transformer-engine extra: $TE_EXTRA"
echo

# Build the paths
CUDNN_PATH="$NVIDIA_PATH/cudnn"
NVTX_INCLUDE="$NVIDIA_PATH/nvtx/include"
CUDNN_INCLUDE="$NVIDIA_PATH/cudnn/include"

# Build LD_LIBRARY_PATH entries
LD_LIBS=""
for pkg in "$NVIDIA_PATH"/*/lib; do
    if [ -d "$pkg" ]; then
        [ -n "$LD_LIBS" ] && LD_LIBS="$LD_LIBS:"
        LD_LIBS="$LD_LIBS$pkg"
    fi
done

# Generate the shell config block
CONFIG_BLOCK="# Transformer Engine - NVIDIA pip libraries
export CUDNN_PATH=\"$CUDNN_PATH\"
export C_INCLUDE_PATH=\"$NVTX_INCLUDE:$CUDNN_INCLUDE:\$C_INCLUDE_PATH\"
export CPLUS_INCLUDE_PATH=\"$NVTX_INCLUDE:$CUDNN_INCLUDE:\$CPLUS_INCLUDE_PATH\"
export LD_LIBRARY_PATH=\"$LD_LIBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\""

echo "Add the following to your ~/.bashrc or ~/.zshrc:"
echo
echo "----------------------------------------"
echo "$CONFIG_BLOCK"
echo "----------------------------------------"
echo

# Detect shell config file
if [ -n "$ZSH_VERSION" ] || [ "$SHELL" = "/bin/zsh" ]; then
    RC_FILE="$HOME/.zshrc"
else
    RC_FILE="$HOME/.bashrc"
fi

read -p "Add to $RC_FILE automatically? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "" >> "$RC_FILE"
    echo "$CONFIG_BLOCK" >> "$RC_FILE"
    echo "Added to $RC_FILE"
    echo "Run: source $RC_FILE"
else
    echo "Skipped. Copy the block above manually."
fi

echo
echo "After sourcing, install transformer-engine with:"
echo "  uv sync --extra $TE_EXTRA"
echo
echo "Or manually with pip:"
echo "  pip install --no-build-isolation 'transformer_engine[pytorch,core_cu${CUDA_MAJOR}]'"
