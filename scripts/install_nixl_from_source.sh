#!/bin/bash
set -euo pipefail

# Build UCX 1.19.x with CUDA + IB support, then build NIXL against it.
# Installs the unrepaired wheel (no auditwheel) so NIXL uses the UCX libs
# directly from third_party/ucx/ at runtime via LD_LIBRARY_PATH.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_BIN="$PROJECT_DIR/.venv/bin"
PYTHON="$VENV_BIN/python"

# UCX's --with-verbs silently disables IB/RoCE support when the rdma-core *dev*
# headers (verbs.h / rdma_cma.h) are missing, yielding a TCP-only build. On hosts
# with RDMA devices, require the headers up front instead of failing at runtime.
HAVE_RDMA=0
if compgen -G "/sys/class/infiniband/*" > /dev/null; then
    HAVE_RDMA=1
    if [ ! -f /usr/include/infiniband/verbs.h ] || [ ! -f /usr/include/rdma/rdma_cma.h ]; then
        echo "ERROR: host has RDMA devices but the rdma-core dev headers are missing." >&2
        echo "Install them first, e.g.: apt-get install -y libibverbs-dev librdmacm-dev" >&2
        exit 1
    fi
fi

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
        --with-rdmacm \
        --with-cuda="$CUDA_PATH" \
        --with-ze=no
    make -j"$NPROC"
    make install
    echo "=== UCX installed to $UCX_INSTALL ==="

    # Fail loudly if the IB transports didn't make it into the build.
    if [ "$HAVE_RDMA" = 1 ]; then
        ucx_transports=$(LD_LIBRARY_PATH="$UCX_INSTALL/lib:$UCX_INSTALL/lib/ucx:${LD_LIBRARY_PATH:-}" \
            "$UCX_INSTALL/bin/ucx_info" -d)
        if ! grep -q "Transport: rc_verbs" <<< "$ucx_transports"; then
            echo "ERROR: UCX built without IB (rc_verbs) transport despite RDMA devices being present." >&2
            echo "Check that libibverbs-dev / librdmacm-dev were present at configure time." >&2
            exit 1
        fi
    fi
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
