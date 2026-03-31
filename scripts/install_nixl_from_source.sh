#!/bin/bash
set -euo pipefail

# Build UCX 1.19.x with CUDA + IB support, then build NIXL against it.
# Installs the unrepaired wheel (no auditwheel) so NIXL uses the UCX libs
# directly from third_party/ucx/ at runtime via LD_LIBRARY_PATH.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_BIN="$PROJECT_DIR/.venv/bin"
PYTHON="$VENV_BIN/python"

WORKSPACE="$PROJECT_DIR/nixl_workspace"
mkdir -p "$WORKSPACE"
UCX_SRC="$WORKSPACE/ucx_source"
UCX_INSTALL="$PROJECT_DIR/third_party/ucx"
NIXL_SRC="$WORKSPACE/nixl_source"
NIXL_VERSION="${NIXL_VERSION:-0.10.1}"
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}"
NPROC=$(nproc)

export PATH="$VENV_BIN:$PATH"

echo "=== Building UCX 1.19.x with CUDA + IB ==="
if [ ! -d "$UCX_SRC" ]; then
    git clone https://github.com/openucx/ucx.git "$UCX_SRC"
fi
cd "$UCX_SRC"
git checkout v1.19.x

if [ ! -f "$UCX_INSTALL/lib/libucs.so" ]; then
    ./autogen.sh
    ./configure \
        --prefix="$UCX_INSTALL" \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --enable-mt \
        --with-verbs \
        --with-cuda="$CUDA_PATH" \
        --with-ze=no
    make -j"$NPROC"
    make install
    echo "=== UCX installed to $UCX_INSTALL ==="
else
    echo "=== UCX already built, skipping ==="
fi

echo "=== Building NIXL $NIXL_VERSION ==="
if [ ! -d "$NIXL_SRC" ]; then
    git clone https://github.com/ai-dynamo/nixl.git "$NIXL_SRC"
else
    cd "$NIXL_SRC" && git fetch --tags
fi
cd "$NIXL_SRC"
git checkout "$NIXL_VERSION"

export PKG_CONFIG_PATH="$UCX_INSTALL/lib/pkgconfig"
export LD_LIBRARY_PATH="$UCX_INSTALL/lib:$UCX_INSTALL/lib/ucx:${LD_LIBRARY_PATH:-}"

# Build and install directly (no auditwheel) so NIXL links to our UCX at runtime
WHEEL_DIR="$PROJECT_DIR/deps"
mkdir -p "$WHEEL_DIR"
uv pip install pip 2>/dev/null
"$PYTHON" -m pip wheel . --no-deps --wheel-dir="$WHEEL_DIR"

WHEEL=$(ls "$WHEEL_DIR"/nixl*.whl | head -1)
echo "=== NIXL wheel built at: $WHEEL ==="
