#!/usr/bin/env bash
# Build the Model Express metadata server (+ fetch a Redis binary) for elastic inference.
#
# The Python client (`modelexpress` extra, used by the vLLM loaders) comes from PyPI;
# this script only provides the two server-side binaries the SLURM control-plane node
# runs: `modelexpress-server` (Rust, built with cargo) and `redis-server` (its metadata
# backend). Binaries are installed to third_party/modelexpress/bin.
#
# Prerequisites: Rust toolchain (rustup), protoc.
#
# Usage:
#   bash scripts/install_modelexpress.sh
#
# Options:
#   --ref REF    modelexpress git tag/commit (default: v0.4.0, matches the PyPI client pin)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MX_GIT_REPO="https://github.com/ai-dynamo/modelexpress.git"
MX_GIT_REF="v0.4.0"
REDIS_VERSION="7.4.2"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ref) MX_GIT_REF="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

BIN_DIR="$REPO_ROOT/third_party/modelexpress/bin"
mkdir -p "$BIN_DIR"

# --- modelexpress-server (Rust) ---
if [[ -x "$BIN_DIR/modelexpress-server" ]] && "$BIN_DIR/modelexpress-server" --version 2>/dev/null | grep -q "${MX_GIT_REF#v}"; then
    echo "modelexpress-server ${MX_GIT_REF} already installed at $BIN_DIR"
else
    command -v cargo >/dev/null || { echo "cargo not found — install Rust via rustup first" >&2; exit 1; }
    BUILD_DIR=$(mktemp -d)
    trap 'rm -rf "$BUILD_DIR"' EXIT
    echo "Cloning modelexpress @ $MX_GIT_REF"
    git clone --depth 1 --branch "$MX_GIT_REF" "$MX_GIT_REPO" "$BUILD_DIR/modelexpress"
    echo "Building modelexpress-server (this takes a few minutes)"
    (cd "$BUILD_DIR/modelexpress" && cargo build --release --bin modelexpress-server)
    cp "$BUILD_DIR/modelexpress/target/release/modelexpress-server" "$BIN_DIR/"
    echo "Installed modelexpress-server to $BIN_DIR"
fi

# --- redis-server (metadata backend) ---
if [[ -x "$BIN_DIR/redis-server" ]]; then
    echo "redis-server already installed at $BIN_DIR"
elif command -v redis-server >/dev/null; then
    ln -sf "$(command -v redis-server)" "$BIN_DIR/redis-server"
    echo "Symlinked system redis-server into $BIN_DIR"
else
    BUILD_DIR="${BUILD_DIR:-$(mktemp -d)}"
    echo "Building redis-server $REDIS_VERSION from source"
    curl -fsSL "https://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz" | tar -xz -C "$BUILD_DIR"
    (cd "$BUILD_DIR/redis-${REDIS_VERSION}" && make -j redis-server MALLOC=libc >/dev/null)
    cp "$BUILD_DIR/redis-${REDIS_VERSION}/src/redis-server" "$BIN_DIR/"
    echo "Installed redis-server to $BIN_DIR"
fi

echo "Done. Binaries in $BIN_DIR:"
ls -la "$BIN_DIR"
